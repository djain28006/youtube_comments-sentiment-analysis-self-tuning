import torch
import numpy as np
import re
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from googleapiclient.discovery import build
from collections import Counter

EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness",
    "surprise", "neutral"
]

class YouTubeSentimentPipeline:
    """
    End-to-end pipeline:
    YouTube URL → Comments → RoBERTa → Emotions → Insights
    """

    def __init__(self, model_path: str, developer_key: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
        self.model = RobertaForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval() 
        self.youtube = build(
            "youtube",
            "v3",
            developerKey=developer_key
        )

   
    def _extract_video_id(self, url: str) -> str:
        match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
        return match.group(1) if match else None

    def _fetch_comments(self, video_id: str, max_comments: int):
        comments = []
        request = self.youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            textFormat="plainText"
        )

        while request and len(comments) < max_comments:
            response = request.execute()
            for item in response["items"]:
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(comment)

                if len(comments) >= max_comments:
                    break

            request = self.youtube.commentThreads().list_next(
                request, response
            )

        return comments

    def _predict_emotions(self, comments, threshold: float):
        emotion_counter = Counter()
        supporting_comments = []

        for text in comments:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            ).to(self.device)

            with torch.no_grad():  
                outputs = self.model(**inputs)

            logits = outputs.logits.squeeze()
            probs = torch.sigmoid(logits).cpu().numpy()

            predicted_indices = np.where(probs > threshold)[0]

            for idx in predicted_indices:
                emotion = EMOTION_LABELS[idx]
                emotion_counter[emotion] += 1
                supporting_comments.append(text)

        return emotion_counter, supporting_comments

    def _generate_insight(self, dominant_emotions):
        if not dominant_emotions:
            return "Audience response is mixed with no dominant emotional trend."

        if "joy" in dominant_emotions or "admiration" in dominant_emotions:
            return "The audience reaction is largely positive, indicating strong appreciation and enjoyment."

        if "anger" in dominant_emotions or "disappointment" in dominant_emotions:
            return "The video triggered dissatisfaction or negative sentiment among viewers."

        return "The video evokes diverse emotional responses, suggesting nuanced audience engagement."


    def analyze_youtube_video(self, url: str, max_comments=20, threshold=0.05):
        video_id = self._extract_video_id(url)

        if not video_id:
            return {"error": "Invalid YouTube URL"}

        comments = self._fetch_comments(video_id, max_comments)

        if not comments:
            return {"error": "No comments found"}

        emotion_counter, supporting_comments = self._predict_emotions(
            comments, threshold
        )

        dominant_emotions = [
            emotion for emotion, _ in emotion_counter.most_common(5)
        ]

        return {
            "total_comments_analyzed": len(comments),
            "dominant_emotions": dominant_emotions,
            "top_supporting_comments": supporting_comments[:5],
            "video_review": self._generate_insight(dominant_emotions)
        }
