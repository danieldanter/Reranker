import requests
import json
from typing import List, Dict, Any
import PyPDF2
import io

class AzureLLMClient:
    def __init__(self):
        self.config = {
            'domain': '506triggereu',
            'key': '5959556c551749d0b2c5807ea77919df',
            'deployment_name': 'gpt-4.1-mini',
            'api_version': '2024-02-15-preview'
        }
        self.base_url = f"https://{self.config['domain']}.openai.azure.com"
        self.headers = {
            "Content-Type": "application/json",
            "api-key": self.config['key']
        }
        
        # GPT-4.1-mini can handle 1,047,576 tokens!
        # Leave room for prompt and response, use 900k tokens for input
        # Roughly 3 chars per token = 2.7 million characters
        self.max_input_chars = 2_700_000  # 2.7 million chars!
    
    def generate_questions_simple(self, full_text: str, num_questions: int = 20) -> List[Dict]:
        """With 1M+ token limit, we can send almost any PDF in one go!"""
        
        text_length = len(full_text)
        print(f"PDF text length: {text_length:,} characters")
        
        if text_length <= self.max_input_chars:
            # 99% of PDFs will fit in one call!
            print(f"Entire PDF fits in one API call! Generating {num_questions} questions...")
            return self._generate_questions_single_call(full_text, num_questions)
        else:
            # Only for REALLY huge documents (rare)
            print(f"PDF is enormous ({text_length:,} chars > {self.max_input_chars:,}), splitting...")
            
            # Just split in half
            mid_point = text_length // 2
            part1 = full_text[:mid_point]
            part2 = full_text[mid_point:]
            
            questions_per_part = (num_questions // 2) + 2
            
            print("Processing first half...")
            q1 = self._generate_questions_single_call(part1, questions_per_part)
            
            print("Processing second half...")
            q2 = self._generate_questions_single_call(part2, questions_per_part)
            
            all_questions = q1 + q2
            
            # Remove duplicates and return requested number
            seen = set()
            unique_questions = []
            for q in all_questions:
                q_text = q.get('question', '').lower().strip()
                if q_text not in seen:
                    seen.add(q_text)
                    unique_questions.append(q)
            
            return unique_questions[:num_questions]
    
    def _generate_questions_single_call(self, text: str, num_questions: int) -> List[Dict]:
        """Single API call with our comprehensive prompt - handles huge texts!"""
        
        # Our detailed prompt from before
        prompt = f"""Als Experte für Bildungsevaluation, erstelle {num_questions} vielfältige und anspruchsvolle Frage-Antwort-Paare basierend auf dem folgenden Text.

FRAGENTYPEN DIE DU ERSTELLEN MUSST (mische alle Typen):

1. **Faktische Fragen** (30%):
   - Direkte Informationsabfrage (Wer, Was, Wann, Wo)
   - Beispiel: "Wer ist der Autor?" → "Daniel Danter"

2. **Verständnisfragen** (25%):
   - Erklärung von Konzepten und Zusammenhängen
   - Beispiel: "Was versteht man unter Sarkopenie?" → "Altersbedingter Verlust von Muskelmasse und -funktion"

3. **Analytische Fragen** (20%):
   - Warum-Fragen, Ursache-Wirkung, Mechanismen
   - Beispiel: "Warum tritt Sarkopenie im Alter auf?" → "Durch verminderte Proteinsynthese und hormonelle Veränderungen"

4. **Synthesefragen** (15%):
   - Verbindung mehrerer Informationen
   - Beispiel: "Wie hängen Ernährung und Training bei Sarkopenie zusammen?" → [Antwort aus Text]

5. **Detailfragen** (10%):
   - Spezifische Zahlen, Daten, Methoden
   - Beispiel: "Welche Stichprobengröße wurde verwendet?" → "n=150"

WICHTIGE REGELN:
- ✅ Alle Fragen und Antworten MÜSSEN auf Deutsch sein
- ✅ Antworten müssen VOLLSTÄNDIG aus dem vorliegenden Text ableitbar sein
- ✅ Variiere die Fragewörter: Wer, Was, Wann, Wo, Warum, Wie, Welche, Wodurch, Wozu, Inwiefern
- ✅ Mische einfache und komplexe Fragen
- ✅ Antworten sollen präzise aber vollständig sein (1-3 Sätze)
- ✅ Erstelle Fragen aus verschiedenen Teilen des Textes (Anfang, Mitte, Ende)
- ❌ KEINE Fragen deren Antwort nicht im Text steht
- ❌ KEINE Ja/Nein Fragen
- ❌ KEINE zu allgemeinen Fragen

SCHWIERIGKEITSGRADE:
- "leicht": Direkt im Text zu finden, ein Suchbegriff reicht
- "mittel": Erfordert Verständnis eines Absatzes oder Zusammenhangs
- "schwer": Erfordert Synthese mehrerer Textstellen oder tieferes Verständnis

JSON FORMAT (exakt einhalten):
{{
  "questions": [
    {{
      "question": "Präzise formulierte Frage auf Deutsch?",
      "answer": "Vollständige Antwort aus dem Text, 1-3 Sätze",
      "type": "faktisch|verständnis|analytisch|synthese|detail",
      "difficulty": "leicht|mittel|schwer",
      "context_needed": "single_sentence|paragraph|multiple_sections"
    }}
  ]
}}

TEXT ZUR ANALYSE (Gesamtes Dokument):
{text}

AUFGABE: Erstelle genau {num_questions} hochwertige Fragen aus dem GESAMTEN Text. Achte darauf, dass die Fragen verschiedene Abschnitte des Dokuments abdecken."""

        url = f"{self.base_url}/openai/deployments/{self.config['deployment_name']}/chat/completions?api-version={self.config['api_version']}"
        
        payload = {
            "messages": [
                {
                    "role": "system", 
                    "content": """Du bist ein Experte für die Erstellung von Evaluationsfragen für RAG-Systeme. 
                    Du hast Zugriff auf den KOMPLETTEN Text und sollst Fragen aus allen Bereichen erstellen."""
                },
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 32768,  # Can output up to 32k tokens!
            "response_format": {"type": "json_object"}
        }
        
        try:
            print(f"Sending {len(text):,} chars to API...")
            response = requests.post(url, headers=self.headers, json=payload, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                data = json.loads(content)
                
                questions = data.get('questions', [])
                print(f"Received {len(questions)} questions from API")
                
                # Validate and format
                validated = []
                for q in questions:
                    if q.get('question') and q.get('answer'):
                        validated.append({
                            'question': q['question'],
                            'answer': q['answer'],
                            'type': q.get('type', 'faktisch'),
                            'difficulty': q.get('difficulty', 'mittel'),
                            'context_needed': q.get('context_needed', 'paragraph')
                        })
                
                return validated
            else:
                print(f"API Error: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            print(f"Error: {e}")
            return []

class PDFProcessor:
    @staticmethod
    def extract_full_text(file) -> str:
        """Extract ALL text from PDF"""
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            full_text = ""
            total_pages = len(pdf_reader.pages)
            
            print(f"Extracting text from {total_pages} pages...")
            
            for page_num in range(total_pages):
                if page_num % 10 == 0:
                    print(f"Processing page {page_num}/{total_pages}...")
                
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                
                if page_text.strip():
                    # Add page marker for context
                    full_text += f"\n[Seite {page_num + 1}]\n"
                    full_text += page_text
            
            print(f"Extracted {len(full_text)} characters from {total_pages} pages")
            return full_text
            
        except Exception as e:
            print(f"Error extracting PDF: {e}")
            return ""