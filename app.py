"""
Medical Chatbot Berbasis Sentence-BERT — versi Streamlit
Dikonversi dari notebook: medical_chatbot_535250044.ipynb

Dataset: gabungan ruslanmv/ai-medical-chatbot dan tejas1206/medical-chatbot (Hugging Face)
"""

import random
import re
import warnings

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# KONFIGURASI HALAMAN
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MedBot - Medical Chatbot",
    page_icon="🩺",
    layout="centered",
)

SAMPLE_SIZE_PER_DATASET = 1500   # jumlah baris yang diambil dari MASING-MASING dataset
FINAL_SAMPLE_SIZE = 1200         # jumlah baris final setelah digabung & dibersihkan
RANDOM_SEED = 42

CATEGORY_KEYWORDS = {
    "diabetes": ["diabetes", "blood sugar", "insulin", "glucose"],
    "hipertensi": ["hypertension", "blood pressure", "high bp"],
    "jantung": ["heart attack", "chest pain", "cardiac", "heart disease", "palpitation"],
    "sakit kepala": ["headache", "migraine"],
    "asma": ["asthma", "wheezing", "inhaler", "shortness of breath"],
    "demam": ["fever", "high temperature"],
    "batuk pilek": ["cough", "cold", "flu", "sore throat", "runny nose"],
    "maag": ["stomach pain", "gastritis", "acid reflux", "ulcer", "abdominal pain"],
    "kolesterol": ["cholesterol", "ldl", "hdl"],
    "kulit": ["rash", "skin", "itching", "acne"],
    "kesehatan mental": ["depression", "anxiety", "stress", "mental health", "panic"],
    "covid": ["covid", "coronavirus"],
    "kehamilan": ["pregnant", "pregnancy"],
    "kesehatan anak": ["baby", "infant", "toddler", "my child"],
}

INDONESIAN_STOPWORDS = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "dengan", "untuk",
    "pada", "adalah", "atau", "juga", "dalam", "tidak", "akan", "ada",
    "saya", "kamu", "anda", "ia", "mereka", "kami", "kita", "bisa",
    "sudah", "bila", "jika", "maka", "oleh", "karena", "apa",
    "bagaimana", "berapa", "kapan", "dimana", "siapa", "apakah", "cara",
    "lebih", "sangat", "dapat", "nya", "pun", "lagi", "belum",
    "telah", "namun", "tapi", "serta", "meski", "agar", "supaya", "hal",
    "the", "is", "are", "was", "what", "how", "why", "when", "where",
}


# ---------------------------------------------------------------------------
# SETUP NLTK & SBERT (cached, hanya dijalankan sekali per sesi server)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Menyiapkan model NLP...")
def setup_nlp():
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)

    lemmatizer = WordNetLemmatizer()
    english_stopwords = set(stopwords.words("english"))
    all_stopwords = INDONESIAN_STOPWORDS | english_stopwords

    use_sbert = True
    sbert_model = None
    try:
        from sentence_transformers import SentenceTransformer
        sbert_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    except Exception:
        use_sbert = False

    return lemmatizer, all_stopwords, sbert_model, use_sbert


def make_preprocess_fn(lemmatizer, all_stopwords):
    from nltk.tokenize import word_tokenize

    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in all_stopwords and len(t) > 2]
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
        return " ".join(tokens)

    return preprocess_text


# ---------------------------------------------------------------------------
# LOAD & GABUNGKAN DATASET (cached)
# ---------------------------------------------------------------------------
def load_hf_dataset_safely(name, n_rows, split_name="train"):
    from datasets import load_dataset

    try:
        ds = load_dataset(name, split=f"{split_name}[:{n_rows}]")
    except Exception:
        ds_full = load_dataset(name)
        split_key = split_name if split_name in ds_full else list(ds_full.keys())[0]
        ds = ds_full[split_key].select(range(min(n_rows, len(ds_full[split_key]))))
    return ds


def standardize_qa(dataset, question_col=None, answer_col=None, source_name="dataset"):
    df_raw = dataset.to_pandas()
    cols = list(df_raw.columns)

    known_pairs = [
        ("Patient", "Doctor"),
        ("question", "answer"),
        ("Question", "Answer"),
        ("input", "output"),
        ("instruction", "response"),
        ("instruction", "output"),
        ("prompt", "completion"),
        ("patient_message", "doctor_response"),
    ]

    if question_col and answer_col:
        q_col, a_col = question_col, answer_col
    else:
        q_col, a_col = None, None
        for q, a in known_pairs:
            if q in cols and a in cols:
                q_col, a_col = q, a
                break

    if q_col is None or a_col is None:
        text_cols = [c for c in cols if df_raw[c].dtype == object]
        if source_name == "medical-chatbot" and len(text_cols) == 1 and text_cols[0] == "text":
            questions, answers = [], []
            for entry in df_raw["text"].astype(str):
                q_text, a_text = "", ""
                patient_idx = entry.lower().find("patient:")
                doctor_idx = entry.lower().find("doctor:")

                if patient_idx != -1 and doctor_idx != -1:
                    if patient_idx < doctor_idx:
                        q_text = entry[patient_idx + len("patient:"):doctor_idx].strip()
                        a_text = entry[doctor_idx + len("doctor:"):].strip()
                    else:
                        a_text = entry[doctor_idx + len("doctor:"):patient_idx].strip()
                        q_text = entry[patient_idx + len("patient:"):].strip()
                elif patient_idx != -1:
                    q_text = entry[patient_idx + len("patient:"):].strip()
                elif doctor_idx != -1:
                    a_text = entry[doctor_idx + len("doctor:"):].strip()
                else:
                    q_text = entry.strip()

                questions.append(q_text)
                answers.append(a_text)

            temp_df = pd.DataFrame({"question": questions, "answer": answers})
            temp_df = temp_df[(temp_df["question"].str.len() > 5) | (temp_df["answer"].str.len() > 5)]

            q_col, a_col = "question", "answer"
            df_raw = temp_df
            cols = ["question", "answer"]
        elif len(text_cols) >= 2:
            q_col, a_col = text_cols[0], text_cols[1]
        else:
            raise ValueError(
                f"[{source_name}] Tidak bisa menemukan kolom question/answer. "
                f"Kolom tersedia: {cols}."
            )

    out = pd.DataFrame(
        {
            "question": df_raw[q_col].astype(str),
            "answer": df_raw[a_col].astype(str),
            "source": source_name,
        }
    )
    return out


def clean_text_field(text, max_len=700):
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^(hi|hello|hey)[\s,!. ]*(doctor|dr\.?)?[\s,!:.-]*", "", text, flags=re.IGNORECASE)
    if len(text) > max_len:
        text = text[:max_len].rsplit(" ", 1)[0] + "..."
    return text.strip()


def detect_category(text):
    text_lower = text.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return category
    return "umum"


@st.cache_data(show_spinner="Memuat & memproses dataset medis...")
def load_and_prepare_dataset():
    ds1 = load_hf_dataset_safely("ruslanmv/ai-medical-chatbot", SAMPLE_SIZE_PER_DATASET)
    ds2 = load_hf_dataset_safely("tejas1206/medical-chatbot", SAMPLE_SIZE_PER_DATASET)

    df1 = standardize_qa(ds1, question_col="Patient", answer_col="Doctor", source_name="ai-medical-chatbot")
    df2 = standardize_qa(ds2, source_name="medical-chatbot")

    df_combined = pd.concat([df1, df2], ignore_index=True)
    df_combined["question"] = df_combined["question"].apply(lambda t: clean_text_field(t, max_len=200))
    df_combined["answer"] = df_combined["answer"].apply(lambda t: clean_text_field(t, max_len=800))

    df_combined = df_combined[df_combined["question"].str.len() > 5]
    df_combined = df_combined[df_combined["answer"].str.len() > 5]
    df_combined = df_combined.drop_duplicates(subset="question").reset_index(drop=True)

    df_combined["category"] = df_combined["question"].apply(detect_category)

    n_categories = df_combined["category"].nunique()
    per_category_cap = max(5, FINAL_SAMPLE_SIZE // n_categories)
    if len(df_combined) > FINAL_SAMPLE_SIZE:
        df_combined = (
            df_combined.groupby("category", group_keys=False)
            .apply(lambda x: x.sample(min(len(x), per_category_cap), random_state=RANDOM_SEED))
            .reset_index(drop=True)
        )

    df = df_combined[["category", "question", "answer"]].sample(
        frac=1, random_state=RANDOM_SEED
    ).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# ENGINE CHATBOT (sama seperti di notebook)
# ---------------------------------------------------------------------------
class MedicalChatbotEngineV3:
    def __init__(self, dataframe, preprocess_fn, sbert_model, use_sbert, threshold=0.35, top_k=3):
        self.df = dataframe
        self.preprocess_fn = preprocess_fn
        self.sbert_model = sbert_model
        self.use_sbert = use_sbert
        self.threshold = threshold if use_sbert else 0.15
        self.top_k = top_k
        self.conversation_history = []

        self._build_index()
        self._define_rules()

    def _build_index(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        if self.use_sbert:
            self.sbert_embeddings = self.sbert_model.encode(
                self.df["question"].tolist(),
                convert_to_tensor=True,
            )

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["processed_question"])

    def _define_rules(self):
        self.rules = {
            "emergency": {
                "patterns": [
                    r"(sesak.*berat|nyeri dada.*berat|tidak.*bernapas|pingsan)",
                    r"(severe chest pain|can.?t breathe|difficulty breathing|unconscious|not breathing|heart attack)",
                ],
                "responses": ["🚨 DARURAT! Hubungi 119 atau segera ke IGD!"],
            },
            "greeting": {
                "patterns": [r"\b(halo|hai|hi|hello|hey|good morning|good afternoon)\b"],
                "responses": ["👋 Halo! Ada yang bisa saya bantu?"],
            },
        }

    def _check_rules(self, text):
        for intent, data in self.rules.items():
            for pattern in data["patterns"]:
                if re.search(pattern, text.lower()):
                    return random.choice(data["responses"])
        return None

    def _search_sbert(self, query):
        from sentence_transformers import util

        emb = self.sbert_model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(emb, self.sbert_embeddings)[0]
        scores = scores.cpu().numpy()
        top_results = np.argsort(-scores)[: self.top_k]
        return [(idx, float(scores[idx])) for idx in top_results]

    def _search_tfidf(self, query):
        from sklearn.metrics.pairwise import cosine_similarity

        processed = self.preprocess_fn(query)
        vec = self.vectorizer.transform([processed])
        scores = cosine_similarity(vec, self.tfidf_matrix).flatten()
        top_results = np.argsort(scores)[::-1][: self.top_k]
        return [(idx, scores[idx]) for idx in top_results]

    def _find_best_match(self, query):
        results = self._search_sbert(query) if self.use_sbert else self._search_tfidf(query)
        best_idx, best_score = results[0]
        return best_idx, best_score

    def _build_context_query(self, user_input):
        if len(self.conversation_history) > 0:
            last_input = self.conversation_history[-1]
            return last_input + " " + user_input
        return user_input

    def get_response(self, user_input):
        if not user_input.strip():
            return "Silakan ketik pertanyaan."

        rule = self._check_rules(user_input)
        if rule:
            return rule

        query = self._build_context_query(user_input)

        if self.use_sbert:
            results = self._search_sbert(query)
            method = "SBERT"
        else:
            results = self._search_tfidf(query)
            method = "TF-IDF"

        best_idx, best_score = results[0]

        if best_score < self.threshold:
            return "🤔 Tidak menemukan jawaban yang cukup relevan."

        boosted = []
        for idx, score in results:
            text = self.df.iloc[idx]["question"]
            bonus = sum(1 for word in user_input.split() if word in text)
            boosted.append((idx, score + 0.05 * bonus))

        best_idx = sorted(boosted, key=lambda x: x[1], reverse=True)[0][0]
        row = self.df.iloc[best_idx]

        self.conversation_history.append(user_input)

        return (
            f"**[Kategori: {row['category']} | {method}]**\n\n"
            f"{row['answer']}\n\n"
            "─────────────────\n"
            "⚠️ Untuk kondisi serius, konsultasikan ke dokter."
        )


# ---------------------------------------------------------------------------
# INISIALISASI (dataset + bot), disimpan di session_state
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Membangun index chatbot...")
def build_bot():
    lemmatizer, all_stopwords, sbert_model, use_sbert = setup_nlp()
    preprocess_fn = make_preprocess_fn(lemmatizer, all_stopwords)

    df = load_and_prepare_dataset()
    df["processed_question"] = df["question"].apply(preprocess_fn)

    bot = MedicalChatbotEngineV3(
        dataframe=df,
        preprocess_fn=preprocess_fn,
        sbert_model=sbert_model,
        use_sbert=use_sbert,
        threshold=0.35,
        top_k=3,
    )
    return bot, df, use_sbert


# ---------------------------------------------------------------------------
# UI STREAMLIT
# ---------------------------------------------------------------------------
def main():
    st.title("🩺 MedBot — Medical Chatbot")
    st.caption(
        "Chatbot medis berbasis Sentence-BERT / TF-IDF. "
        "Dataset: ruslanmv/ai-medical-chatbot + tejas1206/medical-chatbot (Hugging Face)."
    )
    st.warning(
        "⚠️ MedBot bukan pengganti diagnosis dokter. Untuk kondisi serius atau darurat, "
        "segera hubungi layanan medis / IGD.",
        icon="⚠️",
    )

    bot, df, use_sbert = build_bot()

    with st.sidebar:
        st.header("ℹ️ Info")
        st.write(f"Metode pencarian: **{'SBERT' if use_sbert else 'TF-IDF'}**")
        st.write(f"Total Q&A tersimpan: **{len(df)}**")
        st.write(f"Jumlah kategori: **{df['category'].nunique()}**")
        with st.expander("Distribusi kategori"):
            st.bar_chart(df["category"].value_counts())
        if st.button("🔄 Reset percakapan"):
            st.session_state.messages = []
            bot.conversation_history = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "👋 Halo! Saya MedBot. Ada keluhan atau pertanyaan seputar kesehatan?"}
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input("Tulis pertanyaan kesehatan Anda di sini..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("MedBot sedang mengetik..."):
                response = bot.get_response(user_input)
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
