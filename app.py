import os
import torch
import numpy as np
import logging
from typing import Dict, Any, Tuple
import pandas as pd
import re
from datetime import datetime
import pytesseract
from pdf2image import convert_from_path
from transformers import pipeline
import streamlit as st
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Invoice Information Extractor",
    page_icon="📄",
    layout="wide"
)

class FieldValidator:
    """Validator for invoice fields"""

    @staticmethod
    def validate_invoice_number(value: str) -> Tuple[bool, str]:
        """Validate invoice number format"""
        if not value:
            return False, "Empty value"
        if not re.match(r'^[A-Za-z0-9\-/_\.]{3,20}$', value):
            return False, "Invalid format"
        return True, ""

    @staticmethod
    def validate_date(value: str) -> Tuple[bool, str]:
        """Validate and standardize date format"""
        try:
            for fmt in ('%d-%m-%Y', '%d/%m/%Y', '%Y-%m-%d', '%d-%m-%y', '%d/%m/%y'):
                try:
                    date_obj = datetime.strptime(value, fmt)
                    return True, date_obj.strftime('%Y-%m-%d')
                except ValueError:
                    continue
            return False, "Invalid date format"
        except Exception:
            return False, "Invalid date"

    @staticmethod
    def validate_pan(value: str) -> Tuple[bool, str]:
        """Validate PAN number format"""
        if not value:
            return False, "Empty value"
        if not re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]$', value):
            return False, "Invalid PAN format"
        return True, ""

    @staticmethod
    def validate_gstin(value: str) -> Tuple[bool, str]:
        """Validate GSTIN format"""
        if not value:
            return False, "Empty value"
        if not re.match(r'^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$', value):
            return False, "Invalid GSTIN format"
        return True, ""

    @staticmethod
    def validate_amount(value: str) -> Tuple[bool, str]:
        """Validate and standardize amount format"""
        try:
            cleaned = re.sub(r'[^\d.]', '', value)
            amount = float(cleaned)
            if amount <= 0:
                return False, "Amount must be positive"
            return True, f"{amount:.2f}"
        except ValueError:
            return False, "Invalid amount format"

@st.cache_resource
def load_model():
    """Load BERT model with caching"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        bert_pipeline = pipeline(
            "question-answering",
            model="bert-large-uncased-whole-word-masking-finetuned-squad",
            device=0 if device == "cuda" else -1
        )
        return bert_pipeline
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

class InvoiceProcessor:
    def __init__(self):
        """Initialize all required models and processors"""
        self.bert_model = load_model()
        self.validator = FieldValidator()
        self.field_questions = {
            'invoice_number': "What is the invoice number?",
            'invoice_date': "What is the invoice date?",
            'order_id': "What is the order ID or order number?",
            'order_date': "What is the order date?",
            'pan': "What is the PAN number?",
            'gstin': "What is the GSTIN number?",
            'total_amount': "What is the total amount?"
        }

    def _process_with_bert(self, text: str) -> Dict[str, Dict[str, Any]]:
        """Process the text using BERT model"""
        if self.bert_model is None:
            return {}
        
        results = {}
        for field, question in self.field_questions.items():
            try:
                answer = self.bert_model(question=question, context=text)
                results[field] = {
                    'value': answer['answer'],
                    'confidence': answer['score'],
                    'model_source': 'bert'
                }
            except Exception as e:
                logger.error(f"Error processing field {field}: {str(e)}")
                results[field] = {
                    'value': "",
                    'confidence': 0.0,
                    'model_source': 'bert'
                }
        return results

    def _validate_field(self, field_name: str, value: str) -> Tuple[bool, str]:
        """Validate field based on its type"""
        if not value:
            return False, "Empty value"

        if field_name in ['invoice_date', 'order_date']:
            return self.validator.validate_date(value)
        elif field_name == 'pan':
            return self.validator.validate_pan(value)
        elif field_name == 'gstin':
            return self.validator.validate_gstin(value)
        elif field_name == 'total_amount':
            return self.validator.validate_amount(value)
        elif field_name in ['invoice_number', 'order_id']:
            return self.validator.validate_invoice_number(value)

        return True, ""

    def process_invoice(self, uploaded_file) -> Dict[str, Dict[str, Any]]:
        """Process invoice and return structured results with validation"""
        try:
            # Use temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_file_path = tmp_file.name

            # Convert PDF to images
            images = convert_from_path(tmp_file_path)
            first_page = images[0]
            ocr_text = pytesseract.image_to_string(first_page)

            # Process with BERT
            bert_results = self._process_with_bert(ocr_text)
            
            # Validate results
            validated_results = {}
            for field, result in bert_results.items():
                is_valid, message = self._validate_field(field, result['value'])
                validated_results[field] = {
                    'value': result['value'],
                    'confidence': result['confidence'],
                    'model_source': result['model_source'],
                    'is_valid': is_valid,
                    'validation_message': message
                }

            # Clean up temporary file
            os.unlink(tmp_file_path)
            return validated_results
            
        except Exception as e:
            st.error(f"Error processing invoice: {str(e)}")
            return {}

    def convert_to_dataframe(self, results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Convert results to a DataFrame"""
        df_data = {
            'Field': [], 'Value': [], 'Confidence': [], 'Source': [],
            'Valid': [], 'Message': []
        }
        for field, data in results.items():
            df_data['Field'].append(field.replace('_', ' ').title())
            df_data['Value'].append(data['value'])
            df_data['Confidence'].append(f"{data['confidence']:.2%}")
            df_data['Source'].append(data['model_source'].upper())
            df_data['Valid'].append("✅" if data['is_valid'] else "❌")
            df_data['Message'].append(data['validation_message'])
        return pd.DataFrame(df_data)

def main():
    st.title("📄 Invoice Information Extractor")
    st.markdown("Upload a PDF invoice to extract key information using AI")
    
    # Sidebar with info
    with st.sidebar:
        st.header("About")
        st.write("This tool uses BERT AI model to extract:")
        st.write("• Invoice Number")
        st.write("• Invoice Date")
        st.write("• Order ID")
        st.write("• Order Date")
        st.write("• PAN Number")
        st.write("• GSTIN Number")
        st.write("• Total Amount")
        
        st.header("Instructions")
        st.write("1. Upload a PDF invoice")
        st.write("2. Wait for processing")
        st.write("3. Review extracted data")
        st.write("4. Edit if needed")
        st.write("5. Download as Excel")

    # Initialize processor
    if 'processor' not in st.session_state:
        with st.spinner("Loading AI model..."):
            st.session_state.processor = InvoiceProcessor()

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload PDF Invoice", 
        type=['pdf'],
        help="Select a PDF invoice file to extract information"
    )

    if uploaded_file is not None:
        st.success("File uploaded successfully!")
        
        # Process button
        if st.button("🔍 Extract Information", type="primary"):
            with st.spinner("Processing invoice... This may take a few seconds."):
                results = st.session_state.processor.process_invoice(uploaded_file)
                
                if results:
                    # Convert to DataFrame
                    df = st.session_state.processor.convert_to_dataframe(results)
                    
                    # Display results
                    st.subheader("📊 Extracted Information")
                    
                    # Show editable table
                    edited_df = st.data_editor(
                        df, 
                        num_rows="dynamic",
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Confidence": st.column_config.TextColumn("Confidence", width="small"),
                            "Valid": st.column_config.TextColumn("Status", width="small"),
                        }
                    )
                    
                    # Create download data
                    export_data = {}
                    for _, row in edited_df.iterrows():
                        field_key = row['Field'].lower().replace(' ', '_')
                        export_data[field_key] = [row['Value']]
                    
                    export_df = pd.DataFrame(export_data)
                    
                    # Download button
                    st.download_button(
                        label="📥 Download as Excel",
                        data=export_df.to_excel(index=False),
                        file_name="invoice_data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
                    # Show validation summary
                    valid_count = len([row for _, row in edited_df.iterrows() if row['Valid'] == "✅"])
                    total_count = len(edited_df)
                    
                    st.metric("Validation Status", f"{valid_count}/{total_count} fields valid")
                    
                else:
                    st.error("Failed to process the invoice. Please try again.")

    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit & BERT AI | Made by [Your Name]")

if __name__ == "__main__":
    main()
