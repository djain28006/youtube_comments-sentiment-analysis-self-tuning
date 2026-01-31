import torch
import numpy as np
import re
from collections import Counter
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification

class YouTubeScraper:
    def __init__(self, developer_key="AIzaSyCwzrJzwNzfje1xJZmZmBU2L2O-cHWPHk0"):
        self.yt_client = build("youtube", "v3", developerKey=developer_key)

    def extract_video_id(self, url):
        patterns = [
            r"v=([a-zA-Z0-9_-]{11})",
            r"youtu\.be/([a-zA-Z0-9_-]{11})",
            r"embed/([a-zA-Z0-9_-]{11})",
            r"shorts/([a-zA-Z0-9_-]{11})"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    def get_comments(self, video_id, max_results=20):
        comments = []
        next_page_token = None
        
        try:
            while len(comments) < max_results:
                request = self.yt_client.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    textFormat="plainText",
                    maxResults=min(100, max_results - len(comments)), 
                    pageToken=next_page_token,
                    order="relevance"
                )
                response = request.execute()

                for item in response.get("items", []):
                    text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                    comments.append(text)
                
                next_page_token = response.get("nextPageToken")
                if not next_page_token:
                    break
                    
            return comments, None

        except HttpError as e:
            error_msg = f"YouTube API Error: {e.resp.status}"
            content = e.content.decode('utf-8') if hasattr(e, 'content') else str(e)
            
            if "commentsDisabled" in content:
                error_msg = "Comments are disabled for this video."
            elif e.resp.status == 403:
                error_msg += " (Likely invalid API key or quota exceeded)"
            
            print(error_msg)
            return [], error_msg
        except Exception as e:
            error_msg = f"Scraping Error: {e}"
            print(error_msg)
            return [], error_msg


class EmotionClassifier:
    def __init__(self, model_path="roberta-goemotions-model-new"):
        print(f"Loading model from {model_path}...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self.tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
            self.model = RobertaForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

        self.labels = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 
            'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 
            'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 
            'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 
            'relief', 'remorse', 'sadness', 'surprise', 'neutral'
        ]

    def predict(self, text, threshold=0.05):
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits.cpu().detach().numpy()[0]
        scores = 1 / (1 + np.exp(-logits))

        results = {}
        hits = [i for i, score in enumerate(scores) if score > threshold]
        
        if hits:
            non_neutral_hits = [i for i in hits if self.labels[i] != 'neutral']
            if non_neutral_hits:
                for i in non_neutral_hits:
                    results[self.labels[i]] = float(scores[i])
            else:
                results['neutral'] = float(scores[self.labels.index('neutral')])
        else:
            top_idx = np.argmax(scores)
            results[self.labels[top_idx]] = float(scores[top_idx])
            
        return results


class YouTubeSentimentPipeline:
    def __init__(self, model_path="roberta-goemotions-model-new", developer_key="AIzaSyCwzrJzwNzfje1xJZmZmBU2L2O-cHWPHk0"):
        self.scraper = YouTubeScraper(developer_key=developer_key)
        self.classifier = EmotionClassifier(model_path)

    def generate_review(self, emotion_counts, total_comments):
        if total_comments == 0:
            return "Not enough data to generate a review."
            
        meaningful_emotions = {e: c for e, c in emotion_counts.items() if e != 'neutral'}
        if not meaningful_emotions:
            meaningful_emotions = emotion_counts
            
        top_emotions = [e for e, c in Counter(meaningful_emotions).most_common(5)]
        dominant_emotion = top_emotions[0] if top_emotions else "neutral"
        
        pos_set = {"admiration", "amusement", "approval", "caring", "excitement", "gratitude", "joy", "love", "optimism", "pride", "relief"}
        neg_set = {"anger", "annoyance", "disappointment", "disapproval", "disgust", "embarrassment", "fear", "grief", "nervousness", "remorse", "sadness"}
        
        pos_count = sum(emotion_counts[e] for e in pos_set if e in emotion_counts)
        neg_count = sum(emotion_counts[e] for e in neg_set if e in emotion_counts)
        
        review = f"According to the comments, the sentiment of people is as follows:\n\n"
        
        if pos_count > neg_count * 1.5:
            review += f"🌟 **Overwhelmingly Positive Reception**: The audience is vibing with this content! The most prominent sentiment is **{dominant_emotion.upper()}**. Viewers are expressing strong appreciation, likely due to the entertaining or helpful nature of the video."
        elif neg_count > pos_count * 1.5:
             review += f"⚠️ **Critical Audience Reaction**: The feedback indicates distinctive dissatisfaction using **{dominant_emotion.upper()}**. Several viewers are expressing concerns or frustration, suggesting the content might be controversial or technical issues were present."
        else:
             review += f"⚖️ **Mixed or Balanced Views**: The audience is split. While some are showing **{dominant_emotion.upper()}**, there's a complex mix of reactions. This often happens with thought-provoking topics or debates."

        review += f"\n\n**Top Emotional Drivers**: {', '.join([e.capitalize() for e in top_emotions[:3]])}."
        
        return review

    def analyze_youtube_video(self, url, max_comments=20, threshold=0.05):
        video_id = self.scraper.extract_video_id(url)
        if not video_id:
            return {"error": "Invalid YouTube URL"}

        comments, error = self.scraper.get_comments(video_id, max_results=max_comments)
        if error:
            return {"error": error}
        if not comments:
            return {"error": "No comments found (they might be disabled or the video is private)."}

        all_emotions = []
        detailed_results = []
        
        for comment in comments:
            prediction = self.classifier.predict(comment, threshold=threshold)
            all_emotions.extend(list(prediction.keys()))
            detailed_results.append({
                "text": comment,
                "predictions": prediction
            })

        counter = Counter(all_emotions)
        meaningful_counter = Counter({e: c for e, c in counter.items() if e != 'neutral'})
        if not meaningful_counter:
            meaningful_counter = counter
            
        dominant_emotions = meaningful_counter.most_common(5)
        top_labels = [e for e, c in dominant_emotions]
        
        comment_scores = []
        for res in detailed_results:
            max_signal = 0
            for label in top_labels:
                max_signal = max(max_signal, res['predictions'].get(label, 0))
            if max_signal > 0:
                comment_scores.append((res['text'], max_signal))
        
        comment_scores.sort(key=lambda x: x[1], reverse=True)
        top_supporting_comments = [text for text, score in comment_scores[:5]]
        
        review = self.generate_review(counter, len(comments))

        return {
            "video_id": video_id,
            "total_comments_analyzed": len(comments),
            "video_review": review,
            "dominant_emotions": [e.capitalize() for e, c in dominant_emotions],
            "top_supporting_comments": top_supporting_comments
        }
