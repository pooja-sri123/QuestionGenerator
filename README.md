# AdaptIQ — Adaptive Automatic Question Generator

> IV Semester Review-2 Project | Department of Artificial Intelligence and Machine Learning

---

## Project Overview

AdaptIQ is an intelligent, web-based question generation system that automatically creates quiz questions from uploaded study material (PDF or text). It uses Natural Language Processing (NLP) and Machine Learning (ML) to generate questions at different difficulty levels and adapts the quiz difficulty in real-time based on student performance — without requiring any paid API or internet connection.

---

## Problem Statement

Traditional question paper preparation is manual, time-consuming, and does not consider individual student learning levels. Existing systems either require paid APIs or generate generic template-based questions. AdaptIQ solves this by generating real, knowledge-based questions locally using NLP and ML.

---

## Key Features

- Upload PDF or paste study text to generate questions
- Generates 4 question types: MCQ, True/False, Fill-in-the-blank, Short Answer
- Difficulty classification: Easy, Medium, Hard using Random Forest and TF-IDF
- Adaptive Quiz Engine: adjusts difficulty based on student performance streaks
- Bloom's Taxonomy tagging (Remember L1 to Create L6)
- Explanation shown after every answer
- Session analytics, accuracy tracking, and history
- 100% local — no paid API, no internet required during quiz

---

## System Architecture

```
+-------------------+
|   User (Browser)  |
+--------+----------+
         |
         | HTTP Request
         v
+--------+----------+
|  Flask Web Server |  (app.py)
|   localhost:5000  |
+--------+----------+
         |
         +---------------------------+------------------+
         |                           |                  |
         v                           v                  v
+--------+----------+   +-----------+------+   +-------+--------+
| question_generator|   |  adaptive.py     |   |  database.py   |
|      .py          |   |  Difficulty      |   |  SQLite DB     |
|  NLP Engine       |   |  Logic Engine    |   |  Session Store |
+--------+----------+   +-----------+------+   +-------+--------+
         |
         +---------------------------+
         |                           |
         v                           v
+--------+----------+   +-----------+------+
|  TF-IDF Vectorizer|   | Random Forest    |
|  Sentence Ranking |   | Classifier       |
|  (scikit-learn)   |   | Easy/Medium/Hard |
+-------------------+   +------------------+
```

### How Each Component Works

| Component | Role |
|---|---|
| `app.py` | Flask server — handles all API routes, connects frontend to backend |
| `question_generator.py` | Core NLP engine — extracts sentences, ranks them, generates all question types |
| `adaptive.py` | Calculates next difficulty based on accuracy and response time |
| `database.py` | Saves and retrieves quiz session history using SQLite |
| `index.html` | Complete frontend — Upload, Question Bank, Quiz, Analytics pages |
| `models/` | Stores trained Random Forest model and TF-IDF vectorizer using joblib |

---

## Project Structure

```
QuestionGenerator/
│
├── backend/
│   ├── app.py                  # Flask app — all API routes
│   │                           # Routes: /, /upload, /api/generate,
│   │                           #         /submit, /history
│   │
│   ├── question_generator.py   # NLP Question Generation Engine
│   │                           # - clean_text()       : removes noise
│   │                           # - split_sentences()  : splits text
│   │                           # - rank_sentences()   : TF-IDF ranking
│   │                           # - make_definition_question()
│   │                           # - make_purpose_question()
│   │                           # - make_process_question()
│   │                           # - make_fill_blank()
│   │                           # - make_true_false()
│   │                           # - make_short_answer()
│   │                           # - generate_questions() : main entry
│   │
│   ├── adaptive.py             # Adaptive difficulty logic
│   │                           # - get_next_difficulty()
│   │                           # - based on accuracy + avg_time
│   │
│   ├── database.py             # SQLite session management
│   │                           # - init_db()      : create tables
│   │                           # - save_session() : store result
│   │                           # - get_sessions() : fetch history
│   │
│   └── models/                 # Saved ML models (auto-generated)
│       ├── clf.pkl             # Random Forest Classifier
│       └── vec.pkl             # TF-IDF Vectorizer
│
├── frontend/
│   ├── index.html              # Main UI
│   │                           # Pages: Upload & Generate,
│   │                           #        Question Bank,
│   │                           #        Take Quiz,
│   │                           #        Analytics
│   │
│   └── quiz.html               # Quiz interface
│
├── .gitignore                  # Ignores venv, models, uploads, db
└── README.md                   # Project documentation
```

---

## Technologies Used

### Frontend
| Technology | Purpose |
|---|---|
| HTML5 | Structure and layout |
| CSS3 | Styling and dark theme UI |
| Vanilla JavaScript | Quiz engine, adaptive logic, API calls |
| PDF.js (cdnjs) | Extracts text from PDF in the browser |

### Backend
| Technology | Purpose |
|---|---|
| Python 3 | Core programming language |
| Flask | Web framework and REST API |
| Flask-CORS | Handles cross-origin requests |

### NLP and Machine Learning
| Technology | Purpose |
|---|---|
| scikit-learn | TF-IDF vectorizer + Random Forest classifier |
| PyMuPDF (fitz) | Extracts text from PDF on the server |
| python-pptx | Extracts text from PowerPoint files |
| joblib | Saves and loads trained ML models |
| Regex (re) | Sentence parsing and question template filling |

### Database and Storage
| Technology | Purpose |
|---|---|
| SQLite | Stores quiz sessions, scores, accuracy, timestamps |
| Python sqlite3 | Built-in database operations |

### Development Tools
| Tool | Purpose |
|---|---|
| Visual Studio Code | IDE |
| Git | Version control |
| GitHub | Remote repository |
| Python venv | Virtual environment |

---

## NLP Question Generation Pipeline

```
Input Text / PDF
      |
      v
1. clean_text()
   - Remove slide numbers, REVIEW labels, section headers
   - Strip emoji bullets, noise patterns

      |
      v
2. split_sentences()
   - Split on punctuation boundaries
   - Filter metadata, author lines, very short lines
   - Split em-dash separated heading+body

      |
      v
3. rank_sentences()  [TF-IDF]
   - Vectorize all sentences
   - Score by total TF-IDF weight
   - Return top-N most informative sentences

      |
      v
4. Question Generation (per sentence)
   |
   +---> make_definition_question()   "What is X?"
   |     Pattern: "X is/are/means Y"
   |
   +---> make_purpose_question()      "What is X used for?"
   |     Pattern: "X is used for/to Y"
   |
   +---> make_process_question()      "How does X function?"
   |     Pattern: sentences with is used / enables / provides
   |
   +---> make_fill_blank()            "___ is used for Y"
   |     Replaces key term with blank
   |
   +---> make_true_false()            "True or False: X"
   |     Converts factual statements
   |
   +---> make_short_answer()          "Explain the role of X"
         Open-ended explanation question

      |
      v
5. Random Forest Classifier
   - Assigns Easy / Medium / Hard
   - Based on sentence length, clause count, jargon density

      |
      v
6. Bloom's Taxonomy Detection
   - Keyword matching against L1-L6 signal words
   - Remember / Understand / Apply / Analyze / Evaluate / Create

      |
      v
Output: List of Question Objects
{ question, type, difficulty, bloom, options, answer }
```

---

## Adaptive Assessment Engine

```
Start Quiz
    |
    v
Choose Difficulty: Easy / Medium / Hard / All / Adaptive
    |
    v
[Adaptive Mode]
    |
    +-- Answer Correct --> correctStreak++
    |       |
    |       +--> correctStreak == 2 --> Difficulty UP (Easy > Medium > Hard)
    |                                   Reset streak, show banner
    |
    +-- Answer Wrong  --> wrongStreak++
            |
            +--> wrongStreak == 2  --> Difficulty DOWN (Hard > Medium > Easy)
                                       Reset streak, show banner

After each answer:
    - Show explanation box with correct answer
    - User clicks Next to continue
    - Progress bar updates

End of Quiz:
    - Show score, accuracy, time
    - Show difficulty journey (e.g. Q3: easy→medium, Q6: medium→hard)
    - Save session to SQLite database
```

---

## API Routes

| Method | Route | Description |
|---|---|---|
| GET | `/` | Serves the main frontend page |
| POST | `/upload` | Uploads PDF, extracts text, returns basic questions |
| POST | `/api/generate` | Main route — generates questions using NLP engine |
| POST | `/submit` | Saves quiz session, returns next recommended difficulty |
| GET | `/history` | Returns all past quiz sessions from database |

---

## Installation and Setup

### Prerequisites
- Python 3.8 or above
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/pooja-sri123/QuestionGenerator.git
cd QuestionGenerator

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install flask flask-cors scikit-learn pymupdf python-pptx joblib

# 4. Run the application
cd backend
python app.py
```

### 5. Open in browser
```
http://localhost:5000
```

---

## How to Use

1. Go to **Upload and Generate** page
2. Paste your study text or upload a PDF
3. Select difficulty mode and question type
4. Click **Generate Questions**
5. Go to **Question Bank** to review generated questions
6. Go to **Take Quiz**, select quiz difficulty, click **Start Quiz**
7. Answer questions — explanation shown after each answer
8. View your performance in **Analytics**

---

## Future Enhancements

- Support for multiple languages
- Export questions to PDF or Word
- Student login and progress tracking
- Diagrammatic question support
- Online deployment using cloud hosting

---

## License

This project is developed for academic purposes as part of the IV Semester curriculum.
