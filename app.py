import logging
import socket
from flask import Flask, request, jsonify
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import random  # For simulation of selection; replace with real logic if needed

# Download VADER lexicon for sentiment analysis
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

app = Flask(__name__)  # Define app here since it's not imported

logger = logging.getLogger(__name__)

@app.route('/', methods=['GET'])
def default_route():
    return 'Python Template'

def analyze_sentiment(title):
    """Enhanced sentiment analysis with crypto-specific boosts."""
    score = sia.polarity_scores(title)['compound']
    
    # Boost for positive crypto keywords
    positive_keywords = ['buy', 'bull', 'reserve', 'adopt', 'surge', 'rally', 'up']
    negative_keywords = ['sell', 'bear', 'crash', 'dump', 'down', 'ban', 'hack']
    
    title_lower = title.lower()
    if any(word in title_lower for word in positive_keywords):
        score += 0.2
    if any(word in title_lower for word in negative_keywords):
        score -= 0.2
    
    if score > 0.1:
        return "POSITIVE", abs(score)
    elif score < -0.1:
        return "NEGATIVE", abs(score)
    return "NEUTRAL", 0

def detect_candle_pattern(candles):
    """Detect candlestick patterns with trend analysis."""
    if len(candles) < 2:
        return "NEUTRAL", 0
    
    # Calculate overall trend: average close change
    changes = [c['close'] - c['open'] for c in candles]
    avg_change = sum(changes) / len(changes)
    volatility = max(c['high'] - c['low'] for c in candles)
    
    last_candle = candles[-1]
    prev_candle = candles[-2]
    
    # Bullish engulfing
    if last_candle['close'] > last_candle['open'] and prev_candle['close'] < prev_candle['open'] and \
       last_candle['close'] > prev_candle['open'] and last_candle['open'] < prev_candle['close']:
        return "BULLISH", abs(avg_change) + volatility / 1000
    
    # Bearish engulfing
    if last_candle['close'] < last_candle['open'] and prev_candle['close'] > prev_candle['open'] and \
       last_candle['close'] < prev_candle['open'] and last_candle['open'] > prev_candle['close']:
        return "BEARISH", abs(avg_change) + volatility / 1000
    
    # Simple trend
    if avg_change > 0:
        return "BULLISH", abs(avg_change)
    elif avg_change < 0:
        return "BEARISH", abs(avg_change)
    
    return "NEUTRAL", 0

def detect_reversal(observation_candles):
    """Detect if there's a reversal in observation candles (e.g., pump and dump)."""
    if len(observation_candles) < 2:
        return False
    
    first = observation_candles[0]
    last = observation_candles[-1]
    
    # If initial up but then down overall
    if first['close'] > first['open'] and last['close'] < first['close']:
        return True  # Possible dump after pump
    return False

def predict_decision(event):
    """Improved prediction: Combine sentiment, patterns, and reversal detection."""
    sentiment, sent_score = analyze_sentiment(event['title'])
    prev_pattern, prev_strength = detect_candle_pattern(event.get('previous_candles', []))
    obs_pattern, obs_strength = detect_candle_pattern(event.get('observation_candles', []))
    
    score = sent_score + prev_strength + obs_strength
    
    if sentiment == "POSITIVE":
        score += 1
    elif sentiment == "NEGATIVE":
        score -= 1
    
    if prev_pattern == "BULLISH" or obs_pattern == "BULLISH":
        score += 1
    elif prev_pattern == "BEARISH" or obs_pattern == "BEARISH":
        score -= 1
    
    # Adjust for reversal: If positive news but reversal detected, lean SHORT
    if detect_reversal(event.get('observation_candles', [])) and sentiment == "POSITIVE":
        score -= 2  # Penalty for potential pump and dump
    
    return "LONG" if score > 0 else "SHORT", abs(score)

@app.route('/trading-bot', methods=['POST'])
def trading_bot():
    news_events = request.get_json()
    
    # Add confidence score for selection
    events_with_conf = []
    for event in news_events:
        obs_candles = event.get("observation_candles", [])
        total_volume = sum(c['volume'] for c in obs_candles) if obs_candles else 0
        decision, conf = predict_decision(event)
        overall_conf = conf + total_volume / 100  # Combine with volume
        events_with_conf.append({
            "id": event["id"],
            "decision": decision,
            "confidence": overall_conf
        })
    
    # Select top 50 by confidence
    selected = sorted(events_with_conf, key=lambda x: x['confidence'], reverse=True)[:50]
    
    # Prepare output
    decisions = [{"id": e["id"], "decision": e["decision"]} for e in selected]
    
    return jsonify(decisions), 200
    
logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)
