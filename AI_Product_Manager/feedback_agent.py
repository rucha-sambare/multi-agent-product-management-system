from feedback_collection_agent.scrapper import FeedbackCollectionAgent
from feedback_analysis_mgr.feedback_classifier import FeedbackClassifierAgent


class FeedbackAgent:

    def __init__(self):

        self.scraper = FeedbackCollectionAgent()

        self.classifier = FeedbackClassifierAgent()


    def run(self, app_name):

        print("Step 1: Collecting reviews...")

        reviews = self.scraper.collect(app_name)

        print("Step 2: Classifying reviews...")

        classified_reviews = self.classifier.classify(reviews)

        return classified_reviews