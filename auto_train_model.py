import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import time

print("🌐 Initializing Automated OSINT Threat Feed Ingestion...")

# ==========================================
# 1. AUTOMATED DATA INGESTION (0 Rs)
# ==========================================
# Fetch LIVE phishing URLs from OpenPhish (Updated every 12 hours)
print("📥 Contacting OpenPhish Global Threat Database...")
try:
    response = requests.get("https://openphish.com/feed.txt")
    phishing_urls = response.text.splitlines()[:500] # Grabbing the top 500 active threats
    print(f"✅ Successfully ingested {len(phishing_urls)} LIVE phishing payloads.")
except Exception as e:
    print("⚠️ Could not reach OpenPhish. Check network connection.")
    phishing_urls = ['http://secure-update-paypal.com/login', 'http://free-netflix-accounts.tk/auth']

# Create a baseline of known safe domains (Normally you'd pull the Alexa Top 1000 list)
safe_urls = [
    'https://www.google.com', 'https://github.com', 'https://www.wikipedia.org',
    'https://www.amazon.com', 'https://stackoverflow.com', 'https://www.apple.com',
    'https://www.microsoft.com', 'https://www.netflix.com', 'https://www.linkedin.com',
    'https://www.spotify.com'
] * 50 # Multiplying the list to balance the dataset against the 500 phishing links

# Build the dynamic DataFrame
df_phishing = pd.DataFrame({'url': phishing_urls, 'label': 1}) # 1 = BAD
df_safe = pd.DataFrame({'url': safe_urls, 'label': 0})         # 0 = GOOD

df = pd.concat([df_phishing, df_safe]).sample(frac=1).reset_index(drop=True)
print(f"📊 Dynamic Dataset Assembled: {len(df)} total URLs ready for analysis.")

# ==========================================
# 2. NLP FEATURE EXTRACTION
# ==========================================
print("⚙️ Tokenizing URL structures...")
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 5))
X = vectorizer.fit_transform(df['url'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 3. TRAIN THE AI ENGINE
# ==========================================
print("🧠 Training Random Forest Classifier on fresh threat data...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Quick Accuracy Check
accuracy = model.score(X_test, y_test)
print(f"🎯 Model Accuracy on current live data: {accuracy * 100:.2f}%")

# ==========================================
# 4. EXPORT TO DASHBOARD
# ==========================================
joblib.dump(vectorizer, 'url_vectorizer.pkl')
joblib.dump(model, 'url_model.pkl')
print("✅ SUCCESS: 'url_model.pkl' updated with today's threat intelligence.")
