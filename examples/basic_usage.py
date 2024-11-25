from nanopore_detector import EventDetector
from nanopore_detector.config import Config

def main():
    # Create configuration
    config = Config()

    # Initialize detector
    detector = EventDetector(config)

    # Process file
    events_df = detector.process_file(
        file_path="your_data.abf",
        output_path="results/",
        debug=True
    )

    print(f"Detected {len(events_df)} events")

if __name__ == "__main__":
    main()
