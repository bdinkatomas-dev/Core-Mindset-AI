# 🚀 Core Mindset AI — Extreme GPT Engine

Core Mindset AI je vysoce optimalizovaný, od nuly implementovaný jazykový model (LLM) postavený na moderní architektuře **GPT/Llama** v čistém PyTorchi. Tento projekt slouží jako edukační open-source framework ukazující, jak efektivně trénovat vlastní model s kapacitou **~130M parametrů** (odpovídající velikosti GPT-2 Small) na běžně dostupném hardwaru nebo bezplatném Google Colabu.

## 💎 Hlavní Architektonické Přednosti (Extreme GPT Quality)

Model opouští akademické standardy starých transformerů a implementuje techniky, které pohání nejmodernější LLM současnosti (Llama 3, Mistral, GPT-4):

* **RMSNorm (Root Mean Square Normalization):** Nahrazuje klasickou LayerNorm. Výrazně zrychluje propustnost GPU a stabilizuje hluboké sítě tím, že počítá pouze rozptyl.
* **SwiGLU Aktivace:** Pokročilé lineární brány s aktivací SiLU (Swish), které drasticky zvyšují vyjadřovací a logické schopnosti modelu v porovnání s GELU/ReLU.
* **Fused Attention Projections:** Sloučení projekčních matic (Query, Key, Value) do jednoho velkého výpočtu (`qkv_proj`), což maximalizuje paralelizaci a výkon grafické karty.
* **Weight Tying (Svázání vah):** Výstupní vrstva sdílí paměť se vstupními embeddingy, což ušetří desítky milionů zbytečných parametrů na disku bez ztráty inteligence.

## 🛠️ Tréninková Pipeline & Optimalizace

Tréninkový skript `train.py` je plně připraven pro nasazení na **NVIDIA T4 (16GB VRAM)** v prostředí Google Colab:

* **Mixed Precision (AMP FP16):** Dynamické přepínání mezi 16-bit a 32-bit výpočty pro poloviční spotřebu VRAM a bleskový trénink.
* **Cosine Learning Rate Scheduler:** Výuková křivka s plynulým zahříváním (`warmup`) a následným kosinusovým poklesem zabraňuje kolapsu gradientu.
* **Gradient Clipping:** Striktní zastropování explodujících gradientů pro bezpečný trénink hluboké 12-vrstvé sítě.
* **GPT-4 Regex Tokenizer:** Vlastní robustní BPE tokenizér čistící text podle standardů OpenAI.

## 📁 Struktura Projektu

```text
├── data/
│   └── training_data.txt       # Trénovací textový korpus
├── src/
│   ├── __init__.py             # Inicializace Python balíčku
│   ├── tokenizer.py            # Custom BPE/Regex Tokenizer
│   ├── dataset.py              # Cachovaný dataset a DataLoader
│   └── model.py                # Šílená GPT Max architektura (RMSNorm, SwiGLU)
├── checkpoints/                # Automaticky ukládané nejlepší modely
└── train.py                    # Hlavní trénovací loop s AMP a Cosine Schedulerem
