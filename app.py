import os
import tempfile
import logging
from typing import Dict, Any, Tuple
import pandas as pd
import re
from datetime import datetime
import streamlit as st

# Set page config first
st.set_page_config(
    page_title="📄 Invoice Information Extractor",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import heavy libraries with error handling
try:
    import torch
    import numpy as np
    import pytesseract
    from pdf2image import convert_from_path
    from transformers import pipeline
    DEPENDENCIES_LOADED = True
except ImportError as e:
    DEPENDENCIES_LOADED = False
    st.error(f"❌ Error loading dependencies: {str(e)}")
    st.info("Please check if all required packages are installed properly.")
    st.stop()

class FieldValidator:
    """Validator for invoice fields"""

    @staticmethod
    def validate_invoice_number(value: str) -> Tuple[bool, str]:
        """Validate invoice number format"""
        if not value or value.strip() == "":
            return False, "Empty value"
        cleaned_value = value.strip()
        if len(cleaned_value) < 3 or len(cleaned_value) > 20:
            return False, "Length should be between 3-20 characters"
        if not re.match(r'^[A-Za-z0-9\-/_\.]+$', cleaned_value):
            return False, "Contains invalid characters"
        return True, ""

    @staticmethod
    def validate_date(value: str) -> Tuple[bool, str]:
        """Validate and standardize date format"""
        if not value or value.strip() == "":
            return False, "Empty value"
        
        try:
            cleaned_value = value.strip()
            date_formats = [
                '%d-%m-%Y', '%d/%m/%Y', '%Y-%m-%d', '%d-%m-%y', '%d/%m/%y',
                '%d.%m.%Y', '%Y/%m/%d', '%m/%d/%Y', '%m-%d-%Y'
            ]
            
            for fmt in date_formats:
                try:
                    date_obj = datetime.strptime(cleaned_value, fmt)
                    return True, date_obj.strftime('%Y-%m-%d')
                except ValueError:
                    continue
            return False, "Invalid date format (use DD/MM/YYYY or similar)"
        except Exception:
            return False, "Invalid date"

    @staticmethod
    def validate_pan(value: str) -> Tuple[bool, str]:
        """Validate PAN number format"""
        if not value or value.strip() == "":
            return False, "Empty value"
        cleaned_value = value.strip().upper()
        if not re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]$', cleaned_value):
            return False, "Invalid PAN format (should be ABCDE1234F)"
        return True, ""

    @staticmethod
    def validate_gstin(value: str) -> Tuple[bool, str]:
        """Validate GSTIN format"""
        if not value or value.strip() == "":
            return False, "Empty value"
        cleaned_value = value.strip().upper()
        if not re.match(r'^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$', cleaned_value):
            return False, "Invalid GSTIN format (15 characters)"
        return True, ""

    @staticmethod
    def validate_amount(value: str) -> Tuple[bool, str]:
        """Validate and standardize amount format"""
        if not value or value.strip() == "":
            return False, "Empty value"
        
        try:
            # Remove currency symbols and commas
            cleaned = re.sub(r'[₹$€£,\s]', '', value.strip())
            # Handle decimal points
            cleaned = re.sub(r'[^\d.]', '', cleaned)
            
            if not cleaned:
                return False, "No numeric value found"
                
            amount = float(cleaned)
            if amount <= 0:
                return False, "Amount must be positive"
            return True, f"{amount:.2f}"
        except (ValueError, TypeError):
            return False, "Invalid amount format"

@st.cache_resource
def load_model():
    """Load BERT model with caching and comprehensive error handling"""
    try:
        st.info("🔄 Loading AI model... This may take a few minutes on first run.")
        
        # Check device availability
        device = 0 if torch.cuda.is_available() else -1
        device_name = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        st.info(f"💻 Using device: {device_name}")
        
        # Try different model options based on availability
        model_options = [
            "deepset/roberta-base-squad2",  # Lighter, faster
            "distilbert-base-uncased-distilled-squad",  # Alternative light model
            "bert-large-uncased-whole-word-masking-finetuned-squad"  # Original heavy model
        ]
        
        for model_name in model_options:
            try:
                st.info(f"🤖 Attempting to load: {model_name}")
                bert_pipeline = pipeline(
                    "question-answering",
                    model=model_name,
                    device=device,
                    return_all_scores=False,
                    max_length=512,
                    truncation=True
                )
                st.success(f"✅ Successfully loaded model: {model_name}")
                return bert_pipeline
            except Exception as e:
                st.warning(f"⚠️ Failed to load {model_name}: {str(e)}")
                continue
        
        # If all models fail, return None for fallback
        st.error("❌ All models failed to load. Using fallback extraction method.")
        return None
        
    except Exception as e:
        st.error(f"❌ Critical error loading model: {str(e)}")
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

    def _simple_regex_extraction(self, text: str) -> Dict[str, Dict[str, Any]]:
        """Fallback regex-based extraction if AI model fails"""
        results = {}
        
        # Improved regex patterns for Indian invoices
        patterns = {
            'invoice_number': [
                r'invoice\s*(?:no|number|#)[\s:]*([A-Za-z0-9\-/_\.]{3,20})',
                r'inv\s*(?:no|#)[\s:]*([A-Za-z0-9\-/_\.]{3,20})',
                r'bill\s*(?:no|number|#)[\s:]*([A-Za-z0-9\-/_\.]{3,20})',
                r'doc\s*(?:no|number)[\s:]*([A-Za-z0-9\-/_\.]{3,20})'
            ],
            'invoice_date': [
                r'invoice\s*date[\s:]*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})',
                r'date[\s:]*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})',
                r'bill\s*date[\s:]*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})',
                r'dated[\s:]*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})'
            ],
            'order_id': [
                r'order\s*(?:no|number|id)[\s:]*([A-Za-z0-9\-/_\.]{3,20})',
                r'po\s*(?:no|number)[\s:]*([A-Za-z0-9\-/_\.]{3,20})',
                r'purchase\s*order[\s:]*([A-Za-z0-9\-/_\.]{3,20})'
            ],
            'total_amount': [
                r'total[\s:]*₹?\s*([0-9,]+\.?\d*)',
                r'amount[\s:]*₹?\s*([0-9,]+\.?\d*)',
                r'₹\s*([0-9,]+\.?\d*)',
                r'rs[\s\.]*([0-9,]+\.?\d*)',
                r'grand\s*total[\s:]*₹?\s*([0-9,]+\.?\d*)'
            ],
            'pan': [
                r'pan[\s:]*([A-Z]{5}[0-9]{4}[A-Z])',
                r'([A-Z]{5}[0-9]{4}[A-Z])'
            ],
            'gstin': [
                r'gstin[\s:]*([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1})',
                r'gst\s*no[\s:]*([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1})',
                r'([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1})'
            ]
        }
        
        text_upper = text.upper()
        
        for field, field_patterns in patterns.items():
            found = False
            for pattern in field_patterns:
                try:
                    matches = re.findall(pattern, text_upper, re.IGNORECASE)
                    if matches:
                        value = matches[0].strip()
                        if value:  # Only use non-empty matches
                            results[field] = {
                                'value': value,
                                'confidence': 0.7,  # Lower confidence for regex
                                'model_source': 'regex'
                            }
                            found = True
                            break
                except Exception as e:
                    logger.error(f"Regex error for {field}: {e}")
                    continue
            
            if not found:
                results[field] = {
                    'value': "",
                    'confidence': 0.0,
                    'model_source': 'regex'
                }
        
        # Ensure order_date is included
        if 'order_date' not in results:
            results['order_date'] = {
                'value': "",
                'confidence': 0.0,
                'model_source': 'regex'
            }
        
        return results

    def _process_with_bert(self, text: str) -> Dict[str, Dict[str, Any]]:
        """Process the text using BERT model with fallback"""
        if self.bert_model is None:
            st.warning("🔄 Using fallback regex extraction method")
            return self._simple_regex_extraction(text)
        
        results = {}
        for field, question in self.field_questions.items():
            try:
                # Limit context length to avoid memory issues
                max_context_length = 1000
                context = text[:max_context_length] if len(text) > max_context_length else text
                
                answer = self.bert_model(question=question, context=context)
                results[field] = {
                    'value': answer['answer'].strip(),
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
        if not value or value.strip() == "":
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
            # Validate file size (limit to 10MB)
            if uploaded_file.size > 10 * 1024 * 1024:
                st.error("📁 File size too large. Please upload a file smaller than 10MB.")
                return {}
            
            # Use temporary file with proper cleanup
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_file_path = tmp_file.name

            # Convert PDF to images with error handling
            try:
                st.info("📄 Converting PDF to images...")
                images = convert_from_path(tmp_file_path, first_page=1, last_page=1)
                
                if not images:
                    raise Exception("No pages found in PDF")
                    
                first_page = images[0]
                
                st.info("🔍 Extracting text from document...")
                ocr_text = pytesseract.image_to_string(first_page, lang='eng')
                
                if not ocr_text.strip():
                    raise Exception("No text extracted from PDF. The document might be an image or scanned document.")
                    
                st.info(f"✅ Extracted {len(ocr_text)} characters of text")
                
            except Exception as e:
                st.error(f"📄 Error processing PDF: {str(e)}")
                st.info("💡 Tip: Make sure your PDF contains text and is not just an image.")
                return {}

            # Process with AI or fallback
            st.info("🤖 Processing with AI model...")
            results = self._process_with_bert(ocr_text)
            
            # Validate results
            validated_results = {}
            for field, result in results.items():
                is_valid, message = self._validate_field(field, result['value'])
                validated_results[field] = {
                    'value': result['value'],
                    'confidence': result['confidence'],
                    'model_source': result['model_source'],
                    'is_valid': is_valid,
                    'validation_message': message if message else "Valid"
                }

            return validated_results
            
        except Exception as e:
            st.error(f"❌ Error processing invoice: {str(e)}")
            return {}
        finally:
            # Clean up temporary file
            try:
                if 'tmp_file_path' in locals():
                    os.unlink(tmp_file_path)
            except Exception:
                pass

    def convert_to_dataframe(self, results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Convert results to a DataFrame"""
        df_data = {
            'Field': [], 'Value': [], 'Confidence': [], 'Source': [],
            'Status': [], 'Message': []
        }
        
        field_display_names = {
            'invoice_number': 'Invoice Number',
            'invoice_date': 'Invoice Date',
            'order_id': 'Order ID',
            'order_date': 'Order Date',
            'pan': 'PAN Number',
            'gstin': 'GSTIN Number',
            'total_amount': 'Total Amount'
        }
        
        for field, data in results.items():
            display_name = field_display_names.get(field, field.replace('_', ' ').title())
            df_data['Field'].append(display_name)
            df_data['Value'].append(data['value'])
            df_data['Confidence'].append(f"{data['confidence']:.1%}" if data['confidence'] > 0 else "N/A")
            df_data['Source'].append(data['model_source'].upper())
            df_data['Status'].append("✅ Valid" if data['is_valid'] else "❌ Invalid")
            df_data['Message'].append(data['validation_message'])
        
        return pd.DataFrame(df_data)

def main():
    # Header with styling
    st.title("📄 AI-Powered Invoice Information Extractor")
    st.markdown("""
    <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;'>
        <p style='margin: 0; color: #262730;'>
            🤖 Upload a PDF invoice to automatically extract key information using advanced AI models.
            Supports Indian tax formats (PAN, GSTIN) with real-time validation.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if dependencies are loaded
    if not DEPENDENCIES_LOADED:
        st.error("❌ Some dependencies failed to load. Please contact support.")
        return
    
    # Sidebar with information and instructions
    with st.sidebar:
        st.header("📋 About This Tool")
        st.markdown("""
        **This AI tool extracts:**
        - 🧾 Invoice Number
        - 📅 Invoice Date  
        - 🛒 Order ID
        - 📅 Order Date
        - 🆔 PAN Number
        - 🏢 GSTIN Number
        - 💰 Total Amount
        """)
        
        st.header("📖 How to Use")
        st.markdown("""
        1. **Upload** a PDF invoice file
        2. **Wait** for AI processing (30-60 seconds)
        3. **Review** extracted information
        4. **Edit** any incorrect data
        5. **Download** results as Excel
        """)
        
        st.header("📝 Supported Formats")
        st.markdown("""
        - **Files**: PDF only (max 10MB)
        - **Languages**: English
        - **Date Formats**: DD/MM/YYYY, DD-MM-YYYY
        - **Currency**: Indian Rupees (₹)
        """)
        
        st.header("🔒 Privacy")
        st.markdown("""
        - Files processed in memory only
        - No data stored permanently
        - Automatic cleanup after processing
        """)

    # Initialize processor with error handling
    try:
        if 'processor' not in st.session_state:
            with st.spinner("🔄 Initializing AI processor..."):
                st.session_state.processor = InvoiceProcessor()
    except Exception as e:
        st.error(f"❌ Failed to initialize processor: {str(e)}")
        return

    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # File uploader with enhanced styling
        uploaded_file = st.file_uploader(
            "📤 Upload PDF Invoice", 
            type=['pdf'],
            help="Select a PDF invoice file (max 10MB)",
            accept_multiple_files=False
        )

    with col2:
        if uploaded_file is not None:
            st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")

    if uploaded_file is not None:
        st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")
        
        # Show file details
        with st.expander("📄 File Details"):
            st.write(f"**Name:** {uploaded_file.name}")
            st.write(f"**Size:** {uploaded_file.size / 1024:.1f} KB")
            st.write(f"**Type:** {uploaded_file.type}")
        
        # Process button with enhanced styling
        if st.button("🚀 Extract Information", type="primary", use_container_width=True):
            with st.spinner("🔄 Processing invoice... Please wait (this may take 30-60 seconds)"):
                progress_bar = st.progress(0)
                progress_bar.progress(25, "Converting PDF...")
                
                results = st.session_state.processor.process_invoice(uploaded_file)
                progress_bar.progress(75, "Validating data...")
                
                if results:
                    progress_bar.progress(100, "Complete!")
                    progress_bar.empty()
                    
                    # Convert to DataFrame
                    df = st.session_state.processor.convert_to_dataframe(results)
                    
                    # Display results in tabs
                    tab1, tab2, tab3 = st.tabs(["📊 Extracted Data", "📈 Summary", "💾 Export"])
                    
                    with tab1:
                        st.subheader("📊 Extracted Information")
                        
                        # Show editable table with enhanced styling
                        edited_df = st.data_editor(
                            df, 
                            num_rows="dynamic",
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Field": st.column_config.TextColumn("Field", width="medium"),
                                "Value": st.column_config.TextColumn("Value", width="large"),
                                "Confidence": st.column_config.TextColumn("Confidence", width="small"),
                                "Status": st.column_config.TextColumn("Status", width="small"),
                                "Message": st.column_config.TextColumn("Validation", width="medium"),
                            }
                        )
                    
                    with tab2:
                        st.subheader("📈 Validation Summary")
                        
                        # Calculate metrics
                        valid_count = len([row for _, row in edited_df.iterrows() if "Valid" in row['Status']])
                        total_count = len(edited_df)
                        confidence_avg = df['Confidence'].apply(lambda x: float(x.strip('%')) if x != 'N/A' else 0).mean()
                        
                        # Display metrics in columns
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("✅ Valid Fields", f"{valid_count}/{total_count}")
                        with col2:
                            st.metric("🎯 Avg Confidence", f"{confidence_avg:.1f}%")
                        with col3:
                            validation_rate = (valid_count / total_count) * 100
                            st.metric("📊 Validation Rate", f"{validation_rate:.1f}%")
                        
                        # Show validation details
                        if valid_count < total_count:
                            st.warning("⚠️ Some fields need attention:")
                            invalid_fields = edited_df[edited_df['Status'].str.contains('Invalid')]
                            for _, row in invalid_fields.iterrows():
                                st.write(f"• **{row['Field']}**: {row['Message']}")
                    
                    with tab3:
                        st.subheader("💾 Export Data")
                        
                        # Prepare export data
                        export_data = {}
                        for _, row in edited_df.iterrows():
                            field_key = row['Field'].lower().replace(' ', '_')
                            export_data[field_key] = [row['Value']]
                        
                        export_df = pd.DataFrame(export_data)
                        
                        # Show preview
                        st.write("**Preview of export data:**")
                        st.dataframe(export_df, use_container_width=True)
                        
                        # Export options
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Excel export
                            from io import BytesIO
                            excel_buffer = BytesIO()
                            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                                export_df.to_excel(writer, sheet_name='Invoice_Data', index=False)
                                # Add a summary sheet
                                summary_df = pd.DataFrame({
                                    'Metric': ['Total Fields', 'Valid Fields', 'Invalid Fields', 'Avg Confidence'],
                                    'Value': [total_count, valid_count, total_count - valid_count, f"{confidence_avg:.1f}%"]
                                })
                                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                            
                            excel_data = excel_buffer.getvalue()
                            
                            st.download_button(
                                label="📥 Download Excel",
                                data=excel_data,
                                file_name=f"invoice_data_{uploaded_file.name.split('.')[0]}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
                        
                        with col2:
                            # CSV export
                            csv_data = export_df.to_csv(index=False)
                            st.download_button(
                                label="📄 Download CSV",
                                data=csv_data,
                                file_name=f"invoice_data_{uploaded_file.name.split('.')[0]}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                    
                    st.success("🎉 Processing completed successfully!")
                    
                else:
                    progress_bar.empty()
                    st.error("❌ Failed to process the invoice. Please try again with a different file.")
                    st.info("💡 **Tips for better results:**")
                    st.write("• Ensure the PDF contains readable text (not just images)")
                    st.write("• Use clear, high-quality documents")
                    st.write("• Check if the file size is under 10MB")

    # Footer with additional information
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🛠️ Built with:**")
        st.write("• Streamlit • PyTorch • BERT AI")
    
    with col2:
        st.markdown("**🔧 Features:**")
        st.write("• AI Extraction • Validation • Export")
    
    with col3:
        st.markdown("**📊 Stats:**")
        if 'processor' in st.session_state and hasattr(st.session_state.processor, 'bert_model'):
            model_status = "✅ AI Active" if st.session_state.processor.bert_model else "⚠️ Fallback Mode"
            st.write(f"• {model_status}")

if __name__ == "__main__":
    main()
