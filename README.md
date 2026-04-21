# AdaptIQ — Adaptive Automatic Question Generator

> IV Semester Review-2 Project | Department of AI & ML | RV College of Engineering


## Project Overview

AdaptIQ is an intelligent, web-based question generation system that automatically creates quiz questions from uploaded study material (PDF or text). It uses Natural Language Processing (NLP) and Machine Learning (ML) to generate questions at different difficulty levels and adapts the quiz difficulty in real-time based on student performance — without requiring any paid API or internet connection.


## Problem Statement

Traditional question paper preparation is manual, time-consuming, and does not consider individual student learning levels. Existing systems either require paid APIs or generate generic template-based questions. AdaptIQ solves this by generating real, knowledge-based questions locally using NLP and ML.


## Key Features

- Upload PDF or paste study text to generate questions
- Generates 4 question types: MCQ, True/False, Fill-in-the-blank, Short Answer
- Difficulty classification: Easy, Medium, Hard using Random Forest + TF-IDF
- Adaptive Quiz Engine: adjusts difficulty based on student performance streaks
- Bloom's Taxonomy tagging (Remember L1 to Create L6)
- Explanation shown after every answer
- Session analytics, accuracy tracking, and history
- 100% local — no paid API, no internet required during quiz


## System Architecture

## Technologies Used

| Layer | Technology |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| PDF Extraction (client) | PDF.js (cdnjs) |
| Backend Framework | Python 3, Flask, Flask-CORS |
| NLP & ML | scikit-learn (TF-IDF, Random Forest) |
| PDF Extraction (server) | PyMuPDF (fitz) |
| PPTX Extraction | python-pptx |
| Model Persistence | joblib |
| Database | SQLite (sqlite3) |
| IDE | Visual Studio Code |
| Version Control | Git, GitHub |


## NLP Question Generation Pipeline

1. Text is extracted from PDF paragraph by paragraph
2. Sentences are cleaned (slide numbers, headers, noise removed)
3. TF-IDF vectorizer ranks sentences by information density
4. Top sentences are selected as question candidates
5. Regex-based NLP templates generate:
   - Definition questions: What is X?
   - Purpose questions: What is X used for?
   - Process questions: How does X function?
   - Fill-in-the-blank: key term replaced with blank
   - True/False: factual statements converted
   - Short Answer: open-ended explanation questions
6. Random Forest classifier assigns Easy / Medium / Hard
7. Bloom's Taxonomy level assigned by keyword detection


## Adaptive Assessment Engine

- Quiz starts at chosen difficulty (Easy / Medium / Hard / All / Adaptive)
- In Adaptive mode:
  - 2 correct answers in a row ? difficulty increases
  - 2 wrong answers in a row ? difficulty decreases
- Live banner notification on every difficulty change
- Difficulty journey shown in results summary


## Project Structure

## Installation and Setup

### Prerequisites
- Python 3.8 or above
- pip

### Steps

`ash
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
`

### 5. Open in browser

## How to Use

1. Go to **Upload and Generate** page
2. Paste your study text or upload a PDF
3. Select difficulty mode and question type
4. Click **Generate Questions**
5. Go to **Question Bank** to review generated questions
6. Go to **Take Quiz**, select quiz difficulty, click **Start Quiz**
7. Answer questions — explanation shown after each answer
8. View your performance in **Analytics**


## Future Enhancements

- Support for multiple languages
- Export questions to PDF or Word
- Student login and progress tracking
- Diagrammatic question support
- Online deployment using cloud hosting






## License

This project is developed for academic purposes as part of the IV Semester curriculum.
