"""CLI: oilgas ingest | ask | chat | forecast | scenario | download-reports."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid


def _print_turn(turn) -> None:
    print(turn.answer)
    if turn.tool_calls:
        print("\n— вызовы инструментов: " + ", ".join(f"{t['name']}({json.dumps(t['args'], ensure_ascii=False)})"
                                                 for t in turn.tool_calls))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="oilgas", description="AI-агент «Нефтегазовый аналитик»")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="проиндексировать PDF-отчёты из data/reports")
    ing.add_argument("--force", action="store_true", help="переиндексировать всё")

    dl = sub.add_parser("download-reports", help="скачать отчёты (EIA STEO и др.) в data/reports")
    dl.add_argument("--months", type=int, default=3)

    sub.add_parser("snapshot-prices", help="сохранить историю цен в data/prices (офлайн-резерв)")

    a = sub.add_parser("ask", help="задать один вопрос")
    a.add_argument("question")

    sub.add_parser("chat", help="интерактивный диалог в терминале")

    f = sub.add_parser("forecast", help="прогноз цены без LLM")
    f.add_argument("--benchmark", default="brent", choices=["brent", "wti"])
    f.add_argument("--horizon", type=int, default=3)
    f.add_argument("--confidence", type=float, default=0.8)

    sc = sub.add_parser("scenario", help="сценарий шока предложения без LLM")
    sc.add_argument("supply_change_mbd", type=float, help="изменение добычи, млн б/с (минус — сокращение)")
    sc.add_argument("--benchmark", default="brent", choices=["brent", "wti"])
    sc.add_argument("--horizon", type=int, default=6)
    sc.add_argument("--offset", type=float, default=0.0)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    if args.cmd == "ingest":
        from oilgas_analyst.rag.store import ingest_reports

        print(json.dumps(ingest_reports(force=args.force), ensure_ascii=False, indent=2))
    elif args.cmd == "download-reports":
        from oilgas_analyst.rag.download import download_reports

        print(json.dumps(download_reports(months=args.months), ensure_ascii=False, indent=2))
    elif args.cmd == "snapshot-prices":
        from oilgas_analyst.forecasting.data import save_snapshot

        print(json.dumps(save_snapshot(), ensure_ascii=False, indent=2))
    elif args.cmd == "forecast":
        from oilgas_analyst.forecasting.service import run_forecast

        print(run_forecast(args.benchmark, args.horizon, args.confidence)[0])
    elif args.cmd == "scenario":
        from oilgas_analyst.forecasting.service import run_supply_shock

        print(run_supply_shock(args.supply_change_mbd, args.benchmark, args.horizon, args.offset)[0])
    elif args.cmd == "ask":
        from oilgas_analyst.agent import OilGasAnalyst

        _print_turn(OilGasAnalyst().ask(args.question))
    elif args.cmd == "chat":
        from oilgas_analyst.agent import OilGasAnalyst

        analyst, thread = OilGasAnalyst(), str(uuid.uuid4())
        print("Нефтегазовый аналитик. Пустая строка или Ctrl+D — выход.")
        while True:
            try:
                q = input("\n> ").strip()
            except EOFError:
                break
            if not q:
                break
            _print_turn(analyst.ask(q, thread))
    return 0


if __name__ == "__main__":
    sys.exit(main())
