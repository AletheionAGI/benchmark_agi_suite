"""
Benchmark AGI Grounding - LLMs Externos (Claude / Gemini / GPT)

Aplica os mesmos 12 testes do AGI Grounding Benchmark em LLMs externos
(sem pipeline ATIC), comparando desempenho bruto entre providers.

Requisitos no .env:
    ANTHROPIC_API_KEY=sk-ant-...
    OPENAI_API_KEY=sk-...
    GOOGLE_API_KEY=AI...

Uso:
    python scripts/benchmark_external_llms.py
"""

import json
import os
import re
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).parent / "external_llm_results"
ENV_PATHS = [
    Path(__file__).parent.parent / ".env",
    Path(__file__).parent.parent / "atic" / ".env",
]

PASS = "[PASS]"
FAIL = "[FAIL]"
INFO = "[INFO]"
WARN = "[WARN]"

# Modelos default por provider
DEFAULT_MODELS = {
    "claude": "claude-sonnet-4-20250514",
    "gpt": "gpt-4o",
    "gemini": "gemini-2.0-flash",
}

PROVIDER_COLORS = {
    "claude": "#cc785c",
    "gpt": "#74aa9c",
    "gemini": "#4285f4",
}


# ---------------------------------------------------------------------------
# Carregar .env
# ---------------------------------------------------------------------------
def load_env() -> Dict[str, str]:
    """Carrega variaveis de .env (busca em dois locais)."""
    env_vars: Dict[str, str] = {}
    for env_path in ENV_PATHS:
        if env_path.exists():
            print(f"  {INFO} Lendo {env_path}")
            with open(env_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, _, val = line.partition("=")
                        env_vars[key.strip()] = val.strip()
    return env_vars


def detect_providers(env_vars: Dict[str, str]) -> Dict[str, str]:
    """Detecta quais providers tem API key configurada."""
    providers = {}
    key_map = {
        "claude": "ANTHROPIC_API_KEY",
        "gpt": "OPENAI_API_KEY",
        "gemini": "GOOGLE_API_KEY",
    }
    for name, env_key in key_map.items():
        val = env_vars.get(env_key, "") or os.environ.get(env_key, "")
        if val and not val.startswith("your_"):
            providers[name] = val
    return providers


# ---------------------------------------------------------------------------
# Interface LLM abstrata
# ---------------------------------------------------------------------------
@dataclass
class LLMResult:
    """Resposta de um LLM externo."""
    text: str
    model: str
    elapsed: float = 0.0


class ExternalLLM(ABC):
    """Interface para LLMs externos."""

    name: str
    model: str

    @abstractmethod
    def query(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> LLMResult:
        """Envia prompt ao LLM. Retorna LLMResult."""
        ...


# ---------------------------------------------------------------------------
# Provider: Claude (Anthropic)
# ---------------------------------------------------------------------------
class ClaudeLLM(ExternalLLM):
    """Wrapper para Claude via Anthropic SDK."""

    def __init__(self, api_key: str, model: Optional[str] = None):
        import anthropic
        self.name = "claude"
        self.model = model or DEFAULT_MODELS["claude"]
        self.client = anthropic.Anthropic(api_key=api_key)

    def query(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> LLMResult:
        if messages is None:
            messages = [{"role": "user", "content": prompt}]
        t0 = time.time()
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )
        text = "".join(
            b.text for b in resp.content if hasattr(b, "text")
        )
        return LLMResult(text=text, model=resp.model, elapsed=time.time() - t0)


# ---------------------------------------------------------------------------
# Provider: GPT (OpenAI)
# ---------------------------------------------------------------------------
class GPTLLM(ExternalLLM):
    """Wrapper para GPT via OpenAI SDK."""

    def __init__(self, api_key: str, model: Optional[str] = None):
        from openai import OpenAI
        self.name = "gpt"
        self.model = model or DEFAULT_MODELS["gpt"]
        self.client = OpenAI(api_key=api_key)

    def query(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> LLMResult:
        if messages is None:
            messages = [{"role": "user", "content": prompt}]
        t0 = time.time()
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )
        text = resp.choices[0].message.content or ""
        return LLMResult(
            text=text, model=resp.model, elapsed=time.time() - t0,
        )


# ---------------------------------------------------------------------------
# Provider: Gemini (Google)
# ---------------------------------------------------------------------------
class GeminiLLM(ExternalLLM):
    """Wrapper para Gemini via Google GenAI SDK."""

    def __init__(self, api_key: str, model: Optional[str] = None):
        from google import genai
        self.name = "gemini"
        self.model = model or DEFAULT_MODELS["gemini"]
        self.client = genai.Client(api_key=api_key)

    def query(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> LLMResult:
        from google.genai import types
        # Converte messages para formato Gemini contents
        contents = []
        if messages:
            for msg in messages:
                role = "user" if msg["role"] == "user" else "model"
                contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part(text=msg["content"])],
                    )
                )
        else:
            contents = [prompt]

        t0 = time.time()
        resp = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
        )
        text = resp.text or ""
        return LLMResult(
            text=text, model=self.model, elapsed=time.time() - t0,
        )


# ---------------------------------------------------------------------------
# Factory de providers
# ---------------------------------------------------------------------------
def create_provider(name: str, api_key: str) -> ExternalLLM:
    """Cria provider por nome."""
    factories = {
        "claude": ClaudeLLM,
        "gpt": GPTLLM,
        "gemini": GeminiLLM,
    }
    return factories[name](api_key=api_key)


# ---------------------------------------------------------------------------
# Testes (reuso do benchmark AGI Grounding)
# ---------------------------------------------------------------------------
# Importa definicoes do benchmark ATIC
sys.path.insert(0, str(Path(__file__).parent.parent / "atic"))
from scripts.test_agi_grounding_suite import TESTS, CATEGORIES, auto_evaluate


@dataclass
class ProviderTestResult:
    """Resultado de um teste para um provider."""
    test_id: str
    name: str
    category: str
    provider: str
    rounds: List[Dict[str, str]] = field(default_factory=list)
    auto_score: int = 0
    judge_score: int = 0
    judge_reason: str = ""
    elapsed: float = 0.0

    @property
    def final_score(self) -> float:
        return round((self.auto_score + self.judge_score) / 2, 1)


# ---------------------------------------------------------------------------
# Execucao de teste (LLM externo, sem pipeline ATIC)
# ---------------------------------------------------------------------------
def run_test_external(
    llm: ExternalLLM,
    test_def,
) -> List[Dict[str, str]]:
    """Executa teste multi-round num LLM externo."""
    rounds = []
    messages = []

    for prompt in test_def.prompts:
        messages.append({"role": "user", "content": prompt})
        result = llm.query(prompt="", messages=list(messages))
        text = result.text
        rounds.append({"prompt": prompt, "response": text})
        messages.append({"role": "assistant", "content": text})

    return rounds


# ---------------------------------------------------------------------------
# Judge (usa o primeiro provider disponivel)
# ---------------------------------------------------------------------------
def judge_evaluate(
    judge_llm: ExternalLLM,
    test_def,
    rounds: List[Dict[str, str]],
) -> Tuple[int, str]:
    """LLM-as-judge. Retorna (score 0-2, reason)."""
    convo_parts = []
    for i, r in enumerate(rounds):
        convo_parts.append(f"Round {i + 1}:")
        convo_parts.append(f"  User: {r['prompt']}")
        resp = r["response"][:800]
        convo_parts.append(f"  AI: {resp}")
    conversation = "\n".join(convo_parts)

    prompt = (
        f"Voce e um avaliador rigoroso de IA.\n"
        f"Teste: {test_def.name}\n"
        f"Rubrica: {test_def.judge_rubric}\n\n"
        f"Conversa:\n{conversation}\n\n"
        f"Pontue 0, 1 ou 2 de acordo com a rubrica.\n"
        f"Formato OBRIGATORIO: SCORE: N / REASON: explicacao curta"
    )

    try:
        result = judge_llm.query(prompt, max_tokens=200, temperature=0.1)
        text = result.text

        match = re.search(r"SCORE:\s*(\d)", text)
        score = int(match.group(1)) if match else 1
        score = min(max(score, 0), 2)

        reason_match = re.search(r"REASON:\s*(.+)", text, re.DOTALL)
        reason = (
            reason_match.group(1).strip()[:200]
            if reason_match
            else text[:200]
        )
        return score, reason
    except Exception as e:
        return 1, f"judge_error: {e}"


# ---------------------------------------------------------------------------
# Metricas agregadas
# ---------------------------------------------------------------------------
def compute_scores(
    results: List[ProviderTestResult],
    providers: List[str],
) -> Dict[str, Any]:
    """Calcula metricas por provider e categoria."""
    scores: Dict[str, Any] = {}

    for prov in providers:
        prov_results = [r for r in results if r.provider == prov]
        if not prov_results:
            continue

        overall = sum(r.final_score for r in prov_results) / len(prov_results)
        overall_norm = round(overall / 2.0, 3)

        cat_scores = {}
        for cat in CATEGORIES:
            cat_res = [r for r in prov_results if r.category == cat]
            if cat_res:
                cat_mean = sum(r.final_score for r in cat_res) / len(cat_res)
                cat_scores[cat] = round(cat_mean / 2.0, 3)
            else:
                cat_scores[cat] = 0.0

        scores[prov] = {
            "overall": overall_norm,
            "categories": cat_scores,
            "total_time": round(sum(r.elapsed for r in prov_results), 1),
        }

    return scores


# ---------------------------------------------------------------------------
# Graficos
# ---------------------------------------------------------------------------
def generate_graphs(
    results: List[ProviderTestResult],
    scores: Dict[str, Any],
    providers: List[str],
) -> None:
    """Gera 3 graficos comparativos entre providers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import math

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "figure.facecolor": "#1e1e1e",
        "axes.facecolor": "#2d2d2d",
        "text.color": "white",
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
    })

    test_ids = [t.test_id for t in TESTS]
    n_prov = len(providers)

    # --- 01: Barras agrupadas por teste ---
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(test_ids))
    width = 0.7 / n_prov

    for i, prov in enumerate(providers):
        means = []
        for tid in test_ids:
            tid_res = [
                r for r in results
                if r.test_id == tid and r.provider == prov
            ]
            m = (
                sum(r.final_score for r in tid_res) / len(tid_res) / 2.0
                if tid_res else 0
            )
            means.append(m)
        offset = (i - (n_prov - 1) / 2) * width
        color = PROVIDER_COLORS.get(prov, "#888888")
        bars = ax.bar(
            x + offset, means, width,
            label=prov.upper(), color=color, alpha=0.85,
        )
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, h + 0.02,
                    f"{h:.2f}", ha="center", va="bottom",
                    fontsize=6, color="white",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(test_ids, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score (0-1)")
    ax.set_title("AGI Grounding - Comparativo LLMs Externos", fontsize=14)
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "01_providers_bars.png",
        dpi=150, facecolor=fig.get_facecolor(),
    )
    plt.close(fig)

    # --- 02: Radar por categoria ---
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    n_cats = len(CATEGORIES)
    angles = [i * 2 * math.pi / n_cats for i in range(n_cats)]
    angles_closed = angles + [angles[0]]

    for prov in providers:
        if prov not in scores:
            continue
        vals = [scores[prov]["categories"].get(c, 0) for c in CATEGORIES]
        vals_closed = vals + [vals[0]]
        color = PROVIDER_COLORS.get(prov, "#888888")
        ax.plot(
            angles_closed, vals_closed, "o-",
            linewidth=2, color=color, label=prov.upper(),
        )
        ax.fill(angles_closed, vals_closed, alpha=0.1, color=color)

    ax.set_xticks(angles)
    ax.set_xticklabels(
        [c.replace(" ", "\n") for c in CATEGORIES], fontsize=7,
    )
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
    ax.set_title("Categorias por Provider", fontsize=13, pad=20)
    fig.savefig(
        OUTPUT_DIR / "02_radar_providers.png",
        dpi=150, facecolor=fig.get_facecolor(),
    )
    plt.close(fig)

    # --- 03: Ranking horizontal ---
    fig, ax = plt.subplots(figsize=(8, 4))
    sorted_provs = sorted(
        providers,
        key=lambda p: scores.get(p, {}).get("overall", 0),
    )
    y_pos = np.arange(len(sorted_provs))
    overalls = [scores.get(p, {}).get("overall", 0) for p in sorted_provs]
    colors = [PROVIDER_COLORS.get(p, "#888888") for p in sorted_provs]

    bars = ax.barh(y_pos, overalls, color=colors, alpha=0.85, height=0.5)
    for bar, val in zip(bars, overalls):
        ax.text(
            val + 0.02, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=12, color="white",
        )
    ax.set_yticks(y_pos)
    ax.set_yticklabels([p.upper() for p in sorted_provs], fontsize=12)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Score Overall (0-1)")
    ax.set_title("Ranking LLMs Externos", fontsize=14)
    plt.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "03_ranking.png",
        dpi=150, facecolor=fig.get_facecolor(),
    )
    plt.close(fig)

    print(f"  {INFO} Graficos salvos em {OUTPUT_DIR}/")


# ---------------------------------------------------------------------------
# Salvar JSON
# ---------------------------------------------------------------------------
def save_results(
    results: List[ProviderTestResult],
    scores: Dict[str, Any],
    providers: List[str],
) -> None:
    """Salva resultados completos em JSON."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tests_out = []
    for tdef in TESTS:
        per_provider = {}
        for prov in providers:
            prov_res = [
                r for r in results
                if r.test_id == tdef.test_id and r.provider == prov
            ]
            if prov_res:
                r = prov_res[0]
                per_provider[prov] = {
                    "auto_score": r.auto_score,
                    "judge_score": r.judge_score,
                    "final_score": r.final_score,
                    "judge_reason": r.judge_reason,
                    "elapsed": round(r.elapsed, 1),
                }
        tests_out.append({
            "test_id": tdef.test_id,
            "name": tdef.name,
            "category": tdef.category,
            "providers": per_provider,
        })

    output = {
        "timestamp": datetime.now().isoformat(),
        "providers": {
            prov: {
                "model": DEFAULT_MODELS.get(prov, "unknown"),
                "overall": scores.get(prov, {}).get("overall", 0),
                "categories": scores.get(prov, {}).get("categories", {}),
                "total_time": scores.get(prov, {}).get("total_time", 0),
            }
            for prov in providers
        },
        "tests": tests_out,
    }

    path = OUTPUT_DIR / "external_llm_results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"  {INFO} Resultados salvos em {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 60)
    print("  AGI Grounding Benchmark - LLMs Externos")
    print("=" * 60)

    # --- Detectar providers ---
    env_vars = load_env()
    available = detect_providers(env_vars)

    if not available:
        print(f"\n  {FAIL} Nenhuma API key encontrada!")
        print(f"  Configure no .env:")
        print(f"    ANTHROPIC_API_KEY=sk-ant-...")
        print(f"    OPENAI_API_KEY=sk-...")
        print(f"    GOOGLE_API_KEY=AI...")
        return 1

    print(f"\n  {INFO} Providers detectados:")
    for name, _ in available.items():
        print(f"    - {name.upper()} ({DEFAULT_MODELS[name]})")

    # --- Criar providers ---
    llms: Dict[str, ExternalLLM] = {}
    for name, key in available.items():
        try:
            llms[name] = create_provider(name, key)
            print(f"  {PASS} {name.upper()} inicializado")
        except ImportError as e:
            print(f"  {WARN} {name.upper()} SDK nao instalado: {e}")
        except Exception as e:
            print(f"  {FAIL} {name.upper()} erro: {e}")

    if not llms:
        print(f"\n  {FAIL} Nenhum provider disponivel!")
        return 1

    providers = list(llms.keys())
    # Primeiro provider como judge
    judge_llm = llms[providers[0]]
    print(f"\n  {INFO} Judge: {judge_llm.name.upper()} ({judge_llm.model})")
    print(f"  {INFO} Testes: {len(TESTS)}")
    total_calls = len(TESTS) * len(providers) + len(TESTS) * len(providers)
    print(f"  {INFO} Chamadas LLM estimadas: ~{total_calls}")
    print()

    # --- Executar benchmark ---
    all_results: List[ProviderTestResult] = []

    for prov_name, llm in llms.items():
        print(f"\n{'=' * 60}")
        print(f"  PROVIDER: {prov_name.upper()} ({llm.model})")
        print(f"{'=' * 60}")

        for test_def in TESTS:
            print(f"\n  --- {test_def.test_id}: {test_def.name} ---")

            result = ProviderTestResult(
                test_id=test_def.test_id,
                name=test_def.name,
                category=test_def.category,
                provider=prov_name,
            )

            # Execucao
            try:
                t0 = time.time()
                rounds = run_test_external(llm, test_def)
                result.elapsed = time.time() - t0
                result.rounds = rounds

                # Preview da resposta
                preview = rounds[-1]["response"][:120].replace("\n", " ")
                print(f"    Resp: {preview}...")
            except Exception as e:
                print(f"    {FAIL} Erro: {e}")
                rounds = [
                    {"prompt": p, "response": f"[ERRO: {e}]"}
                    for p in test_def.prompts
                ]
                result.rounds = rounds

            # Auto-evaluate
            result.auto_score = auto_evaluate(test_def.test_id, rounds)

            # Judge
            result.judge_score, result.judge_reason = judge_evaluate(
                judge_llm, test_def, rounds,
            )

            print(f"    Auto={result.auto_score}/2 "
                  f"Judge={result.judge_score}/2 "
                  f"Final={result.final_score}/2.0 "
                  f"({result.elapsed:.0f}s)")

            all_results.append(result)

    # --- Resultados ---
    print(f"\n{'=' * 60}")
    print("  RESULTADOS FINAIS")
    print(f"{'=' * 60}")

    scores = compute_scores(all_results, providers)

    for prov in providers:
        if prov not in scores:
            continue
        s = scores[prov]
        print(f"\n  {prov.upper()} ({DEFAULT_MODELS[prov]})")
        print(f"    Overall: {s['overall']:.3f}")
        for cat in CATEGORIES:
            val = s["categories"].get(cat, 0)
            bar = "#" * int(val * 20)
            print(f"    {cat:25s}: {val:.2f}  [{bar}]")
        print(f"    Tempo total: {s['total_time']:.0f}s")

    # Ranking
    ranking = sorted(
        providers,
        key=lambda p: scores.get(p, {}).get("overall", 0),
        reverse=True,
    )
    print(f"\n  Ranking:")
    for i, prov in enumerate(ranking, 1):
        overall = scores.get(prov, {}).get("overall", 0)
        print(f"    {i}. {prov.upper():8s} {overall:.3f}")

    # Graficos + JSON
    generate_graphs(all_results, scores, providers)
    save_results(all_results, scores, providers)

    print(f"\n{'=' * 60}")
    print(f"  {PASS} Benchmark concluido")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
