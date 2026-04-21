from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os, fitz
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from database import init_db, save_session, get_sessions
from adaptive import get_next_difficulty
from question_generator import generate_questions   # ← local NLP engine

app = Flask(__name__, static_folder='../frontend')
CORS(app)
init_db()

os.makedirs("uploads", exist_ok=True)
os.makedirs("models", exist_ok=True)


# ── Difficulty classifier (unchanged) ────────────────────────────────────────

def train_classifier():
    questions = [
        "What is NLP?", "Define machine learning.", "What is Python?",
        "How does a neural network work?", "Explain backpropagation.",
        "Why does overfitting occur?", "Analyze the effect of learning rate.",
    ]
    labels = [0, 0, 0, 1, 1, 2, 2]
    vec = TfidfVectorizer()
    X = vec.fit_transform(questions)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X, labels)
    joblib.dump(clf, "models/clf.pkl")
    joblib.dump(vec, "models/vec.pkl")

def classify(question):
    if not os.path.exists("models/clf.pkl"):
        train_classifier()
    clf = joblib.load("models/clf.pkl")
    vec = joblib.load("models/vec.pkl")
    idx = clf.predict(vec.transform([question]))[0]
    return ["easy", "medium", "hard"][idx]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('../frontend', 'index.html')


@app.route('/upload', methods=['POST'])
def upload():
    """Legacy route: extract text from PDF and return rough questions."""
    file = request.files['file']
    path = os.path.join("uploads", file.filename)
    file.save(path)
    doc = fitz.open(path)
    text = " ".join(p.get_text() for p in doc)
    words = text.split()
    chunks = [" ".join(words[i:i+200]) for i in range(0, min(len(words), 1000), 200)]
    results = []
    for chunk in chunks[:5]:
        results.append({
            "question": f"What does this section explain? ({chunk[:80]}...)",
            "difficulty": classify(chunk[:200])
        })
    return jsonify({"questions": results})


@app.route('/api/generate', methods=['POST'])
def api_generate():
    """
    Main question generation endpoint - 100% local, no API key needed.

    Expects JSON:
        {
          "text":       "<study material — one sentence/bullet per line preferred>",
          "count":      10,
          "difficulty": "adaptive" | "easy" | "medium" | "hard",
          "type":       "Mixed (All Types)" | "MCQ Only" | "True/False Only"
                        | "Fill in the Blank" | "Short Answer",
          "bloom":      "Auto (System Decides)" | "Remember (L1)" | ...
        }

    Returns JSON:
        { "questions": [ { question, type, difficulty, bloom,
                           options, answer }, ... ] }
    """
    try:
        data = request.get_json(force=True)
        text       = data.get('text', '').strip()
        count      = int(data.get('count', 10))
        difficulty = data.get('difficulty', 'adaptive')
        q_type     = data.get('type', 'Mixed (All Types)')
        bloom      = data.get('bloom', 'Auto (System Decides)')

        if not text:
            return jsonify({'error': 'No text provided'}), 400
        if count < 1 or count > 50:
            return jsonify({'error': 'Count must be between 1 and 50'}), 400

        # Normalise line-based input: join short lines into paragraph sentences
        # so each bullet/line from the frontend becomes a separate sentence candidate
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if len(lines) > 3:
            # User pasted line-by-line content — join with '. ' to help sentence splitter
            text = '. '.join(lines)

        questions = generate_questions(
            text=text,
            count=count,
            difficulty_mode=difficulty,
            question_type=q_type,
            bloom_focus=bloom,
        )

        if not questions:
            return jsonify({'error': 'Could not extract enough content. Try adding more study material (at least 5–10 sentences).'}), 422

        return jsonify({'questions': questions})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/submit', methods=['POST'])
def submit():
    d = request.json
    nxt = get_next_difficulty(d['accuracy'], d['avg_time'], d['difficulty'])
    save_session(d['student'], d['score'], d['accuracy'], d['avg_time'], d['difficulty'])
    return jsonify({"next_difficulty": nxt})


@app.route('/history')
def history():
    rows = get_sessions()
    return jsonify([{"id": r[0], "student": r[1], "score": r[2],
                     "accuracy": r[3], "difficulty": r[5], "time": r[6]}
                    for r in rows])


if __name__ == '__main__':
    app.run(debug=True, port=5000)