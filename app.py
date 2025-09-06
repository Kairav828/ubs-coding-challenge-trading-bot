import logging
import socket
import os
from flask import Flask, request, jsonify
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import statistics

app = Flask(__name__)  # Define app here since it's not imported

logger = logging.getLogger(__name__)

# Global for SIA (initialized lazily)
sia = None

@app.before_first_request
def load_nltk():
    global sia
    if sia is None:
        nltk.download('vader_lexicon', quiet=True)
        sia = SentimentIntensityAnalyzer()

@app.route('/', methods=['GET'])
def default_route():
    return 'Python Template'  # Simple health check

def analyze_sentiment(title):
    global sia
    if sia is None:
        load_nltk()  # Ensure loaded
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

def calculate_volatility(candles):
    """Calculate volatility as average high-low range."""
    if not candles:
        return 0.0
    ranges = [(c['high'] - c['low']) / c['open'] for c in candles if c['open'] != 0]
    return statistics.mean(ranges) if ranges else 0.0

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

def detect_reversal(observation_candles):
    """Detect potential reversal (e.g., pump then dump)."""
    if len(observation_candles) < 2:
        return False, 0.0
    
    net_change = observation_candles[-1]['close'] - observation_candles[0]['open']
    max_high = max(c['high'] for c in observation_candles)
    reversal_strength = (max_high - observation_candles[-1]['close']) / max_high if max_high != 0 else 0.0
    
    # Reversal if initial up but ends lower
    if observation_candles[0]['close'] > observation_candles[0]['open'] and net_change < 0:
        return True, reversal_strength
    return False, 0.0

def predict_decision(event):
    sentiment, sent_score = analyze_sentiment(event['title'])
    obs_candles = event.get('observation_candles', [])
    prev_candles = event.get('previous_candles', [])
    
    if len(obs_candles) < 2:
        # Not enough data, default predict SHORT
        return "SHORT", 0

    # Entry price is the close of first observation candle
    entry_price = obs_candles[0]['close']
    exit_price_estimate = obs_candles[-1]['close'] # crude estimate using last candle close

    price_change_relative = (exit_price_estimate - entry_price) / entry_price if entry_price != 0 else 0

    # Sentiment contribution (positive push price up, negative down)
    sentiment_val = 0
    if sentiment == "POSITIVE":
        sentiment_val = 1
    elif sentiment == "NEGATIVE":
        sentiment_val = -1

    # Previous candle trend average change
    prev_net, prev_avg = calculate_trend_strength(prev_candles)

    # Patterns in observation
    pattern, pattern_strength = detect_patterns(obs_candles)

    # Weight components
    score = 0

    score += price_change_relative * 10  # Strongest signal: expected price movement from observation candles
    score += sentiment_val * sent_score * 3
    score += prev_avg * 5
    # Boost/Dampen for patterns
    if pattern in ["BULLISH_ENGULFING", "HAMMER"]:
        score += pattern_strength * 5
    elif pattern in ["BEARISH_ENGULFING", "SHOOTING_STAR"]:
        score -= pattern_strength * 5

    # Reversal penalize if sentiment positive but price dropping
    is_rev, rev_strength = detect_reversal(obs_candles)
    if is_rev and sentiment == "POSITIVE":
        score -= (2 + rev_strength * 5)

    decision = "LONG" if score > 0 else "SHORT"
    confidence = abs(score) + sum(c['volume'] for c in obs_candles) / 100
    return decision, confidence

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
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)  # Debug off for production
