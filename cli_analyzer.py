"""
cli_analyzer.py
Command-Line Interface for Experiment 5: Automated Legal Document Analyzer using GPT-4.
Allows analyzing legal agreements from files (.txt, .pdf), sample contracts,
or interactive text inputs with chunking and export options.
"""

import os
import sys
import argparse
from legal_analyzer import (
    DocumentExtractor,
    DocumentChunker,
    GPT4LegalAnalyzer,
    PromptGenerator,
    LegalAnalysisResult
)

# Ensure proper Unicode display on Windows consoles
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def print_banner():
    banner = """
================================================================================
             24CS2506 - Building Applications using GPT-4
       Experiment 5: Automated Legal Document Analyzer using GPT-4
================================================================================
"""
    print(banner)


def display_analysis(result: LegalAnalysisResult):
    print("\n" + "=" * 50)
    print("OUTPUT (Recommended Format):")
    print("=" * 50 + "\n")
    print(result.to_formatted_text())
    print("\n" + "=" * 50)
    print(f"Document Metrics: Words: {result.word_count} | Estimated Tokens: {result.estimated_tokens}")
    print(f"Engine Used: {result.model_used} | Chunks Analyzed: {result.chunk_count}")
    print("=" * 50 + "\n")


def interactive_mode(analyzer: GPT4LegalAnalyzer):
    print_banner()

    sample_map = {
        "1": ("Employment Agreement", "sample_documents/employment_agreement.txt"),
        "2": ("Mutual Non-Disclosure Agreement (NDA)", "sample_documents/non_disclosure_agreement.txt"),
        "3": ("Master Software Services Agreement", "sample_documents/service_agreement.txt"),
        "4": ("Commercial Lease & Rental Agreement", "sample_documents/rental_agreement.txt"),
        "5": ("Long Commercial Contract (Multi-section Chunking)", "sample_documents/long_commercial_contract.txt")
    }

    while True:
        print("\nSelect an option:")
        print("  [1] Sample: Employment Agreement")
        print("  [2] Sample: Non-Disclosure Agreement (NDA)")
        print("  [3] Sample: Master Software Services Agreement")
        print("  [4] Sample: Commercial Lease & Rental Agreement")
        print("  [5] Sample: Long Commercial Contract (Chunking Demo)")
        print("  [6] Load Custom Document (.txt or .pdf)")
        print("  [7] Paste Legal Text Directly")
        print("  [8] Exit")

        choice = input("\nEnter choice (1-8): ").strip()
        if choice == "8":
            print("\nExiting Legal Document Analyzer. Goodbye!")
            break

        doc_text = ""
        use_chunking = False

        if choice in sample_map:
            doc_name, file_path = sample_map[choice]
            print(f"\nLoading {doc_name} from '{file_path}'...")
            try:
                doc_text = DocumentExtractor.extract_from_file(file_path)
            except Exception as e:
                print(f"Error reading file: {e}")
                continue
            if choice == "5":
                use_chunking = True
        elif choice == "6":
            path = input("Enter path to file (.txt or .pdf): ").strip().strip('"').strip("'")
            try:
                doc_text = DocumentExtractor.extract_from_file(path)
            except Exception as e:
                print(f"Error reading file: {e}")
                continue
        elif choice == "7":
            print("\nPaste your legal document text below (End with an empty line or EOF):")
            lines = []
            while True:
                try:
                    line = input()
                    if not line and lines and not lines[-1]:
                        break
                    lines.append(line)
                except EOFError:
                    break
            doc_text = "\n".join(lines).strip()
        else:
            print("Invalid selection. Please choose 1-8.")
            continue

        if not doc_text:
            print("Error: Document text is empty.")
            continue

        stats = DocumentExtractor.get_stats(doc_text)
        print(f"\nDocument Loaded: {stats['words']} words, ~{stats['estimated_tokens']} estimated tokens.")

        if not use_chunking and stats['characters'] > 2000:
            chk_choice = input("Would you like to analyze using document chunking? (y/n) [default: n]: ").strip().lower()
            use_chunking = (chk_choice == 'y')

        print("\nAnalyzing document using GPT-4 legal pipeline...")
        try:
            result = analyzer.analyze(doc_text, use_chunking=use_chunking)
            display_analysis(result)

            save_choice = input("Do you want to save this analysis to a file? (y/n): ").strip().lower()
            if save_choice == 'y':
                default_name = f"analysis_output_{result.document_type.lower().replace(' ', '_')[:25]}.txt"
                out_path = input(f"Enter output file path [default: {default_name}]: ").strip() or default_name
                with open(out_path, "w", encoding="utf-8") as out_f:
                    out_f.write(result.to_formatted_text())
                print(f"Analysis saved to: {os.path.abspath(out_path)}")
        except Exception as e:
            print(f"Analysis error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Automated Legal Document Analyzer using GPT-4")
    parser.add_argument("--file", "-f", help="Path to legal document file (.txt or .pdf)")
    parser.add_argument("--chunk", action="store_true", help="Enable document chunking for long agreements")
    parser.add_argument("--model", "-m", default="gpt-4o", help="Model name (e.g., gpt-4o, gpt-4, gpt-4o-mini)")
    parser.add_argument("--output", "-o", help="Output file path to save analysis results")
    args = parser.parse_args()

    analyzer = GPT4LegalAnalyzer(default_model=args.model)

    if args.file:
        try:
            doc_text = DocumentExtractor.extract_from_file(args.file)
            result = analyzer.analyze(doc_text, model=args.model, use_chunking=args.chunk)
            display_analysis(result)

            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    f.write(result.to_formatted_text())
                print(f"Saved analysis to: {args.output}")
        except Exception as e:
            print(f"Error processing file: {e}")
            sys.exit(1)
    else:
        interactive_mode(analyzer)


if __name__ == "__main__":
    main()
