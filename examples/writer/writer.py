import os
import openai
from typing import List, Tuple

openai.api_key = os.getenv("OPENAI_API_KEY")

class WiMProcessor:
    def __init__(self, segment_size: int = 1500):  # Increased for better context
        self.segment_size = segment_size
        self.client = openai.OpenAI()

    def split_context(self, context: str) -> List[str]:
        print(f"Splitting context into segments of {self.segment_size} characters...")
        words = context.split()
        segments = []
        current_segment = []
        current_length = 0
        for word in words:
            if current_length + len(word) > self.segment_size:
                segments.append(" ".join(current_segment))
                current_segment = [word]
                current_length = len(word)
            else:
                current_segment.append(word)
                current_length += len(word) + 1  # +1 for space
        if current_segment:
            segments.append(" ".join(current_segment))
        print(f"Context split into {len(segments)} segments")
        return segments

    def generate_margin(self, segment: str, query: str) -> Tuple[str, bool]:
        print(f"\nGenerating margin for segment (length: {len(segment)})...")
        
        prompt = f"""
        Extract information relevant to the query: {query}

        Context:
        {segment}

        Provide the answer in the format: <YES/NO>#<Relevant information>.
        Rules:
        - If you find any relevant information, even partial, start with YES#
        - If no relevant information is found, start with NO#
        - Include any potentially relevant details about architectural achievements
        - Mention specific structures, techniques, or characteristics if present
        """

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        margin = response.choices[0].message.content.strip()
        is_relevant = margin.startswith("YES#")
        
        if is_relevant:
            print("Relevant margin generated")
        else:
            print("Irrelevant or no margin generated")
        
        return margin.split("#", 1)[1] if is_relevant else "", is_relevant

    def process_query(self, context: str, query: str) -> str:
        print(f"Processing query: {query}")
        segments = self.split_context(context)
        relevant_margins = []

        for i, segment in enumerate(segments):
            print(f"\nProcessing segment {i+1}/{len(segments)}")
            margin, is_relevant = self.generate_margin(segment, query)
            if is_relevant:
                relevant_margins.append(f"Segment {i+1}: {margin}")

        print(f"\nAggregated {len(relevant_margins)} relevant margins")

        print("\nGenerating final answer...")
        final_prompt = f"""
        Analyze the following information extracted from a longer text about ancient civilizations:

        {' '.join(relevant_margins)}

        Based on this information, answer the following query:
        {query}

        Provide a comprehensive answer that compares and contrasts the architectural achievements of these civilizations. If there's not enough information about a particular civilization, mention that in your answer.
        """

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": final_prompt}],
            temperature=0
        )

        return response.choices[0].message.content.strip()

# Larger example
if __name__ == "__main__":
    wim = WiMProcessor()

    print("=== Large Example: Multi-hop reasoning across a longer context ===")
    context_large = """
    The Rise and Fall of Ancient Civilizations

    Ancient Egypt:
    The civilization of ancient Egypt thrived for over 3000 years along the Nile River. It was ruled by powerful pharaohs and is famous for its monumental pyramids and intricate hieroglyphics. The Great Pyramid of Giza, built for Pharaoh Khufu around 2560 BCE, stands as a testament to their architectural prowess. Egyptian society was highly stratified, with the pharaoh at the top, followed by nobles, priests, scribes, and then farmers and laborers. The Egyptians were master builders, using sophisticated techniques to construct massive stone structures without the use of complex machinery. Their temples, such as those at Luxor and Karnak, featured massive columns and intricate reliefs depicting religious scenes and historical events.

    Ancient Greece:
    Ancient Greek civilization, which flourished from around 800 BCE to 146 BCE, laid the foundations for Western philosophy, science, and art. The city-state of Athens is credited with developing the world's first democracy under the leadership of Pericles in the 5th century BCE. Greek philosophers like Socrates, Plato, and Aristotle profoundly influenced Western thought. The Parthenon, built in Athens during the Golden Age of Pericles, remains an iconic symbol of classical Greek architecture. Greek architecture is characterized by its use of columns, pediments, and a focus on symmetry and proportion. The Greeks developed three main architectural orders: Doric, Ionic, and Corinthian, each with its distinct style of column and capital. Their temples, such as the Temple of Zeus at Olympia, were designed to be viewed from the outside and often surrounded by colonnades.

    Roman Empire:
    The Roman Empire, at its peak in the 2nd century CE, spanned vast territories from Britain to Egypt. It was founded in 27 BCE when Octavian became the first Roman emperor, taking the name Augustus. The Romans were master engineers, constructing an extensive network of roads, aqueducts, and monumental buildings like the Colosseum. Latin, the language of Rome, became the lingua franca of the Western world. The empire's decline began in the 3rd century CE, culminating in the fall of Rome to Germanic invaders in 476 CE. Roman architecture was heavily influenced by Greek styles but also made significant innovations. The Romans perfected the use of the arch and developed concrete, allowing them to build on a scale never before seen. The Pantheon in Rome, with its massive concrete dome, showcases their engineering skills. Roman architecture also included practical structures like aqueducts, bathhouses, and amphitheaters, demonstrating their focus on public works and urban planning.

    These ancient civilizations, while separated by time and geography, all made significant contributions to human history. Their achievements in art, architecture, philosophy, science, and governance continue to influence our world today. The study of these civilizations provides valuable insights into the development of human society and the cyclical nature of historical processes.
    """

    query_large = "What are the key similarities and differences between the architectural achievements of Ancient Egypt, Greece, and Rome?"
    result_large = wim.process_query(context_large, query_large)
    print(f"\nFinal Answer: {result_large}\n")