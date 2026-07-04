import argparse
from extract import extract_api_info

def main():
    parser = argparse.ArgumentParser(description="Bootstrap a SaaS API integration from its docs.")
    parser.add_argument("--url", required=True, help="URL of the API documentation")
    args = parser.parse_args()         # reads what the user typed after 'python main.py'

    print(f"Reading docs from: {args.url}\n")
    result = extract_api_info(args.url)
    print(result)


if __name__ == "__main__":
    main()
