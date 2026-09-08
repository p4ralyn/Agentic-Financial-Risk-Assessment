"""CLI front end. Usage: python main.py VZ"""
import os
import sys

from graph import build_graph, initial_state


def main() -> int:
    if not os.getenv("GOOGLE_API_KEY"):
        print("GOOGLE_API_KEY is not set. Add it to .env or export it.")
        return 1

    ticker = sys.argv[1] if len(sys.argv) > 1 else "VZ"
    question = "Assess this company's bankruptcy risk against its industry peers."

    try:
        result = build_graph().invoke(
            initial_state(ticker, question),
            config={"configurable": {"thread_id": f"cli-{ticker}"},
                    "recursion_limit": 25},
        )
    except ValueError as exc:
        print(exc)
        return 1

    z, band = result.get("z_score"), result.get("z_band")
    print(f"\n{result.get('company_name')} ({result.get('ticker')}) — "
          f"{result.get('industry_name')}")
    print(f"Altman Z: {z if z is None else round(z, 2)} ({band})\n")
    print(result.get("final_risk_assessment_report") or "No report produced.")

    validation = result.get("validation") or {}
    if validation:
        print(f"\nValidation: {validation.get('final_score')}/100 after "
              f"{result.get('validation_attempts')} attempt(s)")
        for flag in validation.get("logic_flags") or []:
            print(f"  - {flag}")

    for err in result.get("data_errors") or []:
        print(f"  peer data unavailable: {err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
