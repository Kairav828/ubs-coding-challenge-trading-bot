import logging
import socket
from flask import Flask, request, jsonify
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
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
    """Advanced sentiment analysis with crypto-specific keywords."""
    score = sia.polarity_scores(title)['compound']
    
    # Expanded crypto-specific keywords
    positive_keywords = ['adoption', 'reserve', 'bullish', 'partnership', 'upgrade', 'halving', 'etf', 'approval', 'buy', 'surge', 'rally', 'trump', 'strategic', 'establish']
    negative_keywords = ['hack', 'ban', 'crash', 'scam', 'regulation', 'selloff', 'bearish', 'dump', 'fraud', 'collapse', 'warning']
    
    title_lower = title.lower()
    keyword_boost = sum(0.2 for word in positive_keywords if word in title_lower) - \
                    sum(0.2 for word in negative_keywords if word in title_lower)
    
    score += keyword_boost
    
    if score > 0.1:
        return "POSITIVE", abs(score)
    elif score < -0.1:
        return "NEGATIVE", abs(score)
    return "NEUTRAL", 0.0

def calculate_trend_strength(candles):
    """Calculate trend strength: net change and average change."""
    if not candles:
        return 0.0, 0.0
    
    net_change = candles[-1]['close'] - candles[0]['open']
    changes = [(c['close'] - c['open']) for c in candles]
    avg_change = statistics.mean(changes)
    
    return net_change, avg_change

def detect_patterns(candles):
    """Detect additional candlestick patterns for better signals."""
    if len(candles) < 2:
        return "NEUTRAL", 0.0
    
    last = candles[-1]
    prev = candles[-2]
    
    # Bullish patterns
    if last['close'] > last['open'] and last['open'] < prev['close'] and last['close'] > prev['open']:
        pattern = "BULLISH_ENGULFING"
        strength = (last['close'] - last['open']) / last['open']
    # Bearish patterns
    elif last['close'] < last['open'] and last['open'] > prev['close'] and last['close'] < prev['open']:
        pattern = "BEARISH_ENGULFING"
        strength = (last['open'] - last['close']) / last['open']
    # Hammer (bullish reversal)
    elif last['close'] > last['open'] and (last['high'] - last['close']) < (last['close'] - last['open']) and (last['open'] - last['low']) > 2 * (last['close'] - last['open']):
        pattern = "HAMMER"
        strength = (last['open'] - last['low']) / last['open']
    # Shooting star (bearish reversal)
    elif last['close'] < last['open'] and (last['close'] - last['low']) < (last['open'] - last['close']) and (last['high'] - last['open']) > 2 * (last['open'] - last['close']):
        pattern = "SHOOTING_STAR"
        strength = (last['high'] - last['open']) / last['open']
    else:
        pattern = "NEUTRAL"
        strength = 0.0
    
    return pattern, strength

def predict_decision(event):
    """Improved prediction: Focus on observation trend, adjust with sentiment and patterns."""
    sentiment, sent_score = analyze_sentiment(event['title'])
    
    prev_candles = event.get('previous_candles', [])
    obs_candles = event.get('observation_candles', [])
    
    if not obs_candles:
        return "SHORT", 0.0  # Default if no data
    
    # Get trends
    _, prev_avg_change = calculate_trend_strength(prev_candles)
    obs_net_change, obs_avg_change = calculate_trend_strength(obs_candles)
    
    # Detect patterns in observation
    obs_pattern, pattern_strength = detect_patterns(obs_candles)
    
    # Base score from observation trend (key for short-term prediction)
    score = (obs_net_change / obs_candles[0]['open']) * 10 + obs_avg_change * 5 + pattern_strength
    
    # Adjust with previous momentum
    score += prev_avg_change * 2
    
    # Sentiment adjustment
    if sentiment == "POSITIVE":
        score += sent_score * 3
    elif sentiment == "NEGATIVE":
        score -= sent_score * 3
    
    # Reversal detection: If positive sentiment but negative obs trend, amplify SHORT
    if sentiment == "POSITIVE" and obs_net_change < 0:
        score -= 5 + abs(obs_net_change / obs_candles[0]['open']) * 10
    elif sentiment == "NEGATIVE" and obs_net_change > 0:
        score += 5 + abs(obs_net_change / obs_candles[0]['open']) * 10
    
    # Pattern adjustments
    if "BULLISH" in obs_pattern or obs_pattern == "HAMMER":
        score += pattern_strength * 5
    elif "BEARISH" in obs_pattern or obs_pattern == "SHOOTING_STAR":
        score -= pattern_strength * 5
    
    conf = abs(score) + sum(c['volume'] for c in obs_candles) / 50  # Confidence for selection
    
    return "LONG" if score > 0 else "SHORT", conf

@app.route('/trading-bot', methods=['POST'])
def trading_bot():
    news_events = request.get_json()
    
    # Process events
    events_with_conf = []
    for event in news_events:
        decision, conf = predict_decision(event)
        events_with_conf.append({
            "id": event["id"],
            "decision": decision,
            "confidence": conf
        })
    
    # Select top 50 by confidence (prioritize high-impact events)
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
