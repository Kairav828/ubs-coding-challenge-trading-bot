import logging
import socket
from flask import Flask, request, jsonify
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import random  
import statistics

# Download VADER lexicon for sentiment analysis
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

app = Flask(__name__)  # Define app here since it's not imported

logger = logging.getLogger(__name__)

@app.route('/', methods=['GET'])
def default_route():
    return 'Python Template'

def analyze_sentiment(title):
    """Enhanced sentiment analysis with crypto-specific adjustments."""
    score = sia.polarity_scores(title)['compound']
    
    # Crypto-specific keywords for sentiment boost
    positive_keywords = ['adoption', 'reserve', 'bullish', 'partnership', 'upgrade', 'halving', 'etf', 'approval']
    negative_keywords = ['hack', 'ban', 'crash', 'scam', 'regulation', 'selloff', 'bearish', 'dump']
    
    title_lower = title.lower()
    keyword_boost = sum(0.15 for word in positive_keywords if word in title_lower) - \
                    sum(0.15 for word in negative_keywords if word in title_lower)
    
    score += keyword_boost
    
    if score > 0.05:
        return "POSITIVE", abs(score)
    elif score < -0.05:
        return "NEGATIVE", abs(score)
    return "NEUTRAL", 0

def calculate_momentum(candles):
    """Calculate simple momentum: average percentage change."""
    if not candles:
        return 0
    changes = [(c['close'] - c['open']) / c['open'] for c in candles if c['open'] != 0]
    return statistics.mean(changes) if changes else 0

def calculate_volatility(candles):
    """Calculate volatility as average high-low range."""
    if not candles:
        return 0
    ranges = [(c['high'] - c['low']) / c['open'] for c in candles if c['open'] != 0]
    return statistics.mean(ranges) if ranges else 0

def detect_reversal(observation_candles):
    """Detect potential reversal (e.g., pump then dump)."""
    if len(observation_candles) < 2:
        return False, 0
    
    net_change = observation_candles[-1]['close'] - observation_candles[0]['open']
    max_high = max(c['high'] for c in observation_candles)
    reversal_strength = (max_high - observation_candles[-1]['close']) / max_high if max_high != 0 else 0
    
    # Reversal if initial up but ends lower
    if observation_candles[0]['close'] > observation_candles[0]['open'] and net_change < 0:
        return True, reversal_strength
    return False, 0

def simple_price_forecast(observation_candles):
    """Simple linear extrapolation for next 30 min direction."""
    if len(observation_candles) < 2:
        return 0
    
    # Use last few changes to predict direction
    changes = [c['close'] - prev['close'] for prev, c in zip(observation_candles[:-1], observation_candles[1:])]
    avg_change = statistics.mean(changes)
    
    # Forecast direction: positive avg change suggests up
    return avg_change / abs(avg_change) if avg_change != 0 else 0

def predict_decision(event):
    """Advanced prediction combining sentiment, momentum, volatility, reversal, and forecast."""
    sentiment, sent_score = analyze_sentiment(event['title'])
    
    all_candles = event.get('previous_candles', []) + event.get('observation_candles', [])
    
    momentum = calculate_momentum(all_candles)
    volatility = calculate_volatility(all_candles)
    is_reversal, rev_strength = detect_reversal(event.get('observation_candles', []))
    forecast_dir = simple_price_forecast(event.get('observation_candles', []))
    
    score = sent_score + (momentum * 10) + forecast_dir
    
    if sentiment == "POSITIVE":
        score += 1
    elif sentiment == "NEGATIVE":
        score -= 1
    
    # Adjust for reversal in positive news (common pump-dump in crypto)
    if sentiment == "POSITIVE" and is_reversal:
        score -= (2 + rev_strength * 5)  # Strong penalty for detected dump
    
    # High volatility with negative momentum suggests SHORT
    if volatility > 0.01 and momentum < 0:  # Threshold based on sample data
        score -= 1.5
    
    conf = abs(score) + volatility  # Confidence includes volatility for selection
    
    return "LONG" if score > 0 else "SHORT", conf

@app.route('/trading-bot', methods=['POST'])
def trading_bot():
    news_events = request.get_json()
    
    # Process events with prediction and confidence
    events_with_conf = []
    for event in news_events:
        obs_candles = event.get("observation_candles", [])
        total_volume = sum(c['volume'] for c in obs_candles) if obs_candles else 0
        decision, conf = predict_decision(event)
        overall_conf = conf + (total_volume / 100)  # Boost with volume
        events_with_conf.append({
            "id": event["id"],
            "decision": decision,
            "confidence": overall_conf
        })
    
    # Select top 50 by confidence (high impact events)
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
