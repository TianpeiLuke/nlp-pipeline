import os
import re
import numpy as np
import pandas as pd
import csv
from bs4 import BeautifulSoup
import sklearn
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from collections.abc import Callable, Mapping
from typing import List, Tuple, Pattern, Union, Dict, Set, Optional, Annotated, Callable
from abc import ABC, abstractmethod
import re
from bs4 import BeautifulSoup


from abc import ABC, abstractmethod

from .constants import *

import boto3
from botocore.exceptions import ClientError
from urllib.request import urlretrieve
import html as ihtml


import six
#from six.moves.urllib.request import urlretrieve

import warnings
#from bs4 import MarkupResemblesLocatorWarning
warnings.filterwarnings("ignore") #, category=MarkupResemblesLocatorWarning, module='bs4')

# --- Base Processor Classes ---
class Processor(ABC):
    processor_name: str
    function_name_list: List[str]
    
    def __init__(self):
        self.processor_name = 'processor'
        self.function_name_list = []
    
    def get_name(self) -> str:
        return self.processor_name
    
    def __call__(self, input_text):
        return self.process(input_text)
    
    @abstractmethod
    def process(self, input_text):
        pass
    
    # Use the >> operator to compose processors.
    def __rshift__(self, other):
        # If self is already a ComposedProcessor, we merge its processors with 'other'
        if isinstance(self, ComposedProcessor):
            return ComposedProcessor(self.processors + [other])
        return ComposedProcessor([self, other])


class ComposedProcessor(Processor):
    def __init__(self, processors: List[Processor]):
        super().__init__()
        self.processors = processors
        # Set function_name_list to a list of the names of each processor.
        self.function_name_list = [p.get_name() for p in processors]
    
    def process(self, input_text):
        for processor in self.processors:
            input_text = processor(input_text)
        return input_text
    
#=====================================================================================
class IdentityProcessor(Processor):
    '''
    An identity processor return a copy of input message itself
    '''
    def __init__(self):
        self.processor_name = 'identity'
    
    
    def process(self, x):
        return x    
    

#=====================================================================================
# Processor 1: Text Normalization
class TextNormalizationProcessor(Processor):
    def __init__(self):
        super().__init__()
        self.processor_name = "text_normalization_processor"

    def process(self, input_text: str):
        # Basic normalization: trim and lowercase.
        normalized = input_text.strip().lower()
        # Collapse multiple spaces into one.
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized

    
# Processor 1: Text Normalization
class TextUpperProcessor(Processor):
    def __init__(self):
        super().__init__()
        self.processor_name = "text_upper_processor"

    def process(self, input_text: str):
        # Basic normalization: trim and lowercase.
        normalized = input_text.strip().upper()
        # Collapse multiple spaces into one.
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized

#=====================================================================================
# Processor 2: Dialogue Splitting
class DialogueSplitterProcessor(Processor):
    def __init__(self):
        super().__init__()
        self.processor_name = "dialogue_splitter_processor"

    def process(self, input_text: str):
        """
        Splits the dialogue into individual messages based on [bom] and [eom] delimiters.
        Returns:
            List of message strings.
        """
        pattern = r'\[bom\](.*?)\[eom\]'
        raw_messages = re.findall(pattern, input_text, flags=re.DOTALL)
        # Strip extra whitespace from each message
        messages = [msg.strip() for msg in raw_messages if msg.strip()]
        return messages

    
#=====================================================================================
# Processor 3: Dialogue Chunker
class DialogueChunkerProcessor(Processor):
    def __init__(self, 
                 tokenizer, 
                 max_tokens=512, 
                 truncate: bool = False,  # Added truncate parameter
                 max_total_chunks: Optional[int] = 5
                ):
        """
        Args:
            tokenizer: A Hugging Face AutoTokenizer instance.
            max_tokens: Maximum token count per chunk.
        """
        super().__init__()
        self.processor_name = "dialogue_chunker_processor"
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.max_total_chunks = max_total_chunks
        self.truncate = truncate

    def process(self, messages: List[str]):
        """
        Chunks a list of messages into groups such that each chunk's token count (without special tokens)
        does not exceed the max_tokens limit.
        
        Returns:
            List of dialogue chunks (each chunk is a concatenated string of messages).
        """
        chunks = []
        current_chunk = []
        current_tokens = 0
        num_chunks = 0  # Track the number of chunks created

        for msg in messages:
            # Count tokens using the HF AutoTokenizer; avoid adding special tokens here
            token_count = len(self.tokenizer.encode(msg, add_special_tokens=False))
            # If adding this message would exceed limit, save current chunk and start a new one.
            if current_tokens + token_count > self.max_tokens:
                if current_chunk:
                    chunks.append(" ".join(current_chunk).strip())
                    num_chunks += 1 # Increment chunk count
                    if self.max_total_chunks is not None and self.truncate and num_chunks >= self.max_total_chunks:
                        break  # Stop if max_total_chunks is reached
                current_chunk = [msg]
                current_tokens = token_count
            else:
                current_chunk.append(msg)
                current_tokens += token_count
                
            if self.max_total_chunks is not None and self.truncate and num_chunks >= self.max_total_chunks:
                break # Stop if max_total_chunks is reached

        if current_chunk:
            chunks.append(" ".join(current_chunk).strip())
        return chunks
    
    
#====================================================================================
# --- Processor 4: Emoji Remover ---
class EmojiRemoverProcessor(Processor):
    def __init__(self):
        super().__init__()
        self.processor_name = 'emoji_remover_processor'
        # Define a regex pattern that captures common emoji ranges.
        self.emoji_pattern = re.compile(
            "["                                     
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags
            u"\U00002702-\U000027B0"  # dingbats
            u"\U000024C2-\U0001F251" 
            "]+", 
            flags=re.UNICODE
        )

    def process(self, input_text: str) -> str:
        """
        Removes emojis from the input text.
        """
        return self.emoji_pattern.sub('', input_text)

    
#=====================================================================================   
# --- Processor 2: HTML Normalization ---
class HTMLNormalizerProcessor(Processor):
    def __init__(self):
        super().__init__()
        self.processor_name = "html_normalizer_processor"
    
    def process(self, input_text: str) -> str:
        """
        Converts HTML content to raw text using BeautifulSoup.
        """
        # Parse the HTML content
        soup = BeautifulSoup(input_text, "html.parser")
        # Get text content with whitespace normalized.
        raw_text = soup.get_text(separator=" ", strip=True)
        return raw_text
    
    
#=============================================================================================   
class AmazonCSProcessor(Processor):
    '''
    CSEmailProcessor remove the email template from CS Central
    '''
    
    def __init__(self):
        self.processor_name = 'cs_email_removal'
        self.function_handler_list = [self.remove_special_symbols, self.amazon_cs_removal_step, self.sentence_norm]
        self.function_name_list = ['remove_special_symbols', 'amazon_cs_removal_step', 'sentence_norm']
    

    def detect_amazon_cs_central_header(self, message: str) -> Tuple[bool, int, int]:
        '''
        For message from CS Central, (Entry Point == 11), a fixed template is used for all messages. 
        The template looks like 
        
        Hello,  We've been contacted by a customer regarding the order identified below.  
        -------------------- 
        Order#: xxx-xxxxxxx-xxxxxxx 
        Item:  
        Reason: xxxxxx  
        Details: (xxxxxx)   
        -------------------  
        To respond to this customer, please reply to this e-mail or visit your seller account at the following link: 
        URL  Sincerely, Customer Service Amazon.xxx URL
        
        
        We detect if such template is present and return the index of message that lists the details part of message (i.e. listed under 'Details:').
        
        '''
        if not isinstance(message, str):
            raise TypeError("Message input should be of str type !")
        
        # Order#:.*(?=Details:XXXX) =>  XXXX
        RE_AMAZON_CS_HEADER_EN = re.compile(r'''Order\s*(#|Number|number):.*(?=Details:)''', re.I | re.S | re.M)
        RE_AMAZON_CS_HEADER_FR = re.compile(r'''Numéro de commande:.*(?=Détails:)''', re.I | re.S | re.M)
        RE_AMAZON_CS_HEADER_DE = re.compile(r'''Bestellnummer\s*:.*(?=Weitere Angaben:)''', re.I | re.S | re.M)
        RE_AMAZON_CS_HEADER_PT = re.compile(r'''Número do pedido\s*:.*(?=Detalhes:)''', re.I | re.S | re.M)
        RE_AMAZON_CS_HEADER_ES = re.compile(r'''Número de pedido\s*:.*(?=Detalles:)''', re.I | re.S | re.M)
        RE_AMAZON_CS_HEADER_MX = re.compile(r'''Pedido\s*#:.*(?=Detalles:)''', re.I | re.S | re.M)
        RE_AMAZON_CS_HEADER_IT = re.compile(r'''Ordine\s*(#|Numero):.*(?=Dettagli:)''', re.I | re.S | re.M)
        RE_AMAZON_CS_HEADER_JA = re.compile(r'''注文番号\s*:.*(?=詳細:)''', re.I | re.S | re.M)
        RE_AMAZON_CS_HEADER_TR = re.compile(r'''Sipariş numarası\s*:.*(?=Ayrıntılar:)''', re.I | re.S | re.M)
        RE_AMAZON_CS_HEADER_NL = re.compile(r'''Bestelnummer\s*:.*(?=Details:)''', re.I | re.S | re.M)
        RE_AMAZON_CS_HEADER_PL = re.compile(r'''Numer zamówienia\s*:.*(?=Szczegóły:)''', re.I | re.S | re.M)


        
        if RE_AMAZON_CS_HEADER_EN.search(message):
            header_end_index = RE_AMAZON_CS_HEADER_EN.search(message).end()
            return True, header_end_index, len('Details: ')
        elif RE_AMAZON_CS_HEADER_FR.search(message):
            header_end_index = RE_AMAZON_CS_HEADER_FR.search(message).end()
            return True, header_end_index, len('Détails: ')
        elif RE_AMAZON_CS_HEADER_DE.search(message):
            header_end_index = RE_AMAZON_CS_HEADER_DE.search(message).end()
            return True, header_end_index, len('Weitere Angaben: ')
        elif RE_AMAZON_CS_HEADER_PT.search(message):
            header_end_index = RE_AMAZON_CS_HEADER_PT.search(message).end()
            return True, header_end_index, len('Detalhes: ')
        elif RE_AMAZON_CS_HEADER_ES.search(message):
            header_end_index = RE_AMAZON_CS_HEADER_ES.search(message).end()
            return True, header_end_index, len('Detalles: ')
        elif RE_AMAZON_CS_HEADER_MX.search(message):
            header_end_index = RE_AMAZON_CS_HEADER_MX.search(message).end()
            return True, header_end_index, len('Detalles: ')
        elif RE_AMAZON_CS_HEADER_IT.search(message):
            header_end_index = RE_AMAZON_CS_HEADER_IT.search(message).end()
            return True, header_end_index, len('Dettagli: ')
        elif RE_AMAZON_CS_HEADER_JA.search(message):
            header_end_index = RE_AMAZON_CS_HEADER_JA.search(message).end()
            return True, header_end_index, len('詳細: ')
        elif RE_AMAZON_CS_HEADER_TR.search(message):
            header_end_index = RE_AMAZON_CS_HEADER_TR.search(message).end()
            return True, header_end_index, len('Ayrıntılar: ')
        elif RE_AMAZON_CS_HEADER_NL.search(message):
            header_end_index = RE_AMAZON_CS_HEADER_NL.search(message).end()
            return True, header_end_index, len('Details: ')
        elif RE_AMAZON_CS_HEADER_PL.search(message):
            header_end_index = RE_AMAZON_CS_HEADER_PL.search(message).end()
            return True, header_end_index, len('Szczegóły: ')       
        else:
            return False, 0, 0 
        
    def detect_amazon_cs_central_signature(self, message: str) -> Tuple[bool, int]:
        '''
        For message from CS Central, (Entry Point == 11), a fixed template is used for all messages. 
        The template looks like 
        
        Hello,  We've been contacted by a customer regarding the order identified below.  
        -------------------- 
        Order#: xxx-xxxxxxx-xxxxxxx 
        Item:  
        Reason: xxxxxx  
        Details: (xxxxxx)   
        -------------------  
        To respond to this customer, please reply to this e-mail or visit your seller account at the following link: 
        URL  Sincerely, Customer Service Amazon.xxx URL
        
        
        We detect if such template is present and return the index of message that starts with 'To respond to this customer, xxx'
        '''
        if not isinstance(message, str):
            raise TypeError("Message input should be of str type !")  
            
        RE_AMAZON_CS_SIGNATURE_EN = re.compile(r'''To respond to this customer, please reply to this e-mail or visit your seller account at the following link:.* Sincerely.*''', re.I | re.S | re.M)
        RE_AMAZON_CS_SIGNATURE_FR = re.compile(r'''Nous vous remercions de répondre directement à ce client en répondant à cet e-mail ou en vous rendant dans votre compte vendeur:.* Cordialement.*''', re.I | re.S | re.M)
        RE_AMAZON_CS_SIGNATURE_DE = re.compile(r'''Um Ihren Käufer zu kontaktieren, antworten Sie bitte auf diese E-Mail oder besuchen Sie Ihr Verkäuferkonto über folgenden Link:.* Freundliche Grüße.*''', re.I | re.S | re.M)
        RE_AMAZON_CS_SIGNATURE_PT = re.compile(r'''Para entrar em contato com este cliente, acesse sua conta de vendedor pelo link:.* Atenciosamente.*''', re.I | re.S | re.M)
        RE_AMAZON_CS_SIGNATURE_ES = re.compile(r'''Para responder al cliente,  por favor responde a este email o bien visita tu cuenta de vendedor en el siguiente enlace:.* Atentamente.*''', re.I | re.S | re.M)
        RE_AMAZON_CS_SIGNATURE_MX = re.compile(r'''Para enviarle una respuesta al cliente, por favor responde a este correo o visita tu cuenta de vendedor en el siguiente enlace:.* Cordialmente.*''', re.I | re.S | re.M)
        RE_AMAZON_CS_SIGNATURE_IT = re.compile(r'''Per rispondere a questo cliente, risponda a quest'e-mail oppure utilizzi il suo account venditore cliccando sul seguente link:.* Cordiali saluti.*''', re.I | re.S | re.M)
        RE_AMAZON_CS_SIGNATURE_JA = re.compile(r'''購入者様へのご連絡は、直接こちらのメールにご返信いただくか、以下セラーアカウントからご返信ください。.* 今後ともAmazon.co.jp をよろしくお願いいたします.*''', re.I | re.S | re.M)
        RE_AMAZON_CS_SIGNATURE_TR = re.compile(r'''Bu müşteriye cevap vermek için lütfen bu e-postayı yanıtlayın veya aşağıdaki bağlantıdan satıcı hesabınızı ziyaret edin:.* Saygılarımızla.*''', re.I | re.S | re.M)
        RE_AMAZON_CS_SIGNATURE_NL = re.compile(r'''Je kunt de klant antwoorden door op deze e-mail te reageren. Je kunt ook reageren via je verkopersaccount:.* Met vriendelijke groet.*''', re.I | re.S | re.M)
        RE_AMAZON_CS_SIGNATURE_PL = re.compile(r'''Aby skontaktować się z tym klientem, proszę odpowiedzieć na ten e-mail lub odwiedzić swoje konto sprzedawcy pod następującym adresem:.* Z poważaniem.*''', re.I | re.S | re.M)
    
    
        if RE_AMAZON_CS_SIGNATURE_EN.search(message):
            signature_start_index = RE_AMAZON_CS_SIGNATURE_EN.search(message).start()
            return True, signature_start_index
        elif RE_AMAZON_CS_SIGNATURE_FR.search(message):
            signature_start_index = RE_AMAZON_CS_SIGNATURE_FR.search(message).start()
            return True, signature_start_index
        elif RE_AMAZON_CS_SIGNATURE_DE.search(message):
            signature_start_index = RE_AMAZON_CS_SIGNATURE_DE.search(message).start()
            return True, signature_start_index
        elif RE_AMAZON_CS_SIGNATURE_PT.search(message):
            signature_start_index = RE_AMAZON_CS_SIGNATURE_PT.search(message).start()
            return True, signature_start_index
        elif RE_AMAZON_CS_SIGNATURE_ES.search(message):
            signature_start_index = RE_AMAZON_CS_SIGNATURE_ES.search(message).start()
            return True, signature_start_index
        elif RE_AMAZON_CS_SIGNATURE_MX.search(message):
            signature_start_index = RE_AMAZON_CS_SIGNATURE_MX.search(message).start()
            return True, signature_start_index      
        elif RE_AMAZON_CS_SIGNATURE_IT.search(message):
            signature_start_index = RE_AMAZON_CS_SIGNATURE_IT.search(message).start()
            return True, signature_start_index    
        elif RE_AMAZON_CS_SIGNATURE_JA.search(message):
            signature_start_index = RE_AMAZON_CS_SIGNATURE_JA.search(message).start()
            return True, signature_start_index
        elif RE_AMAZON_CS_SIGNATURE_TR.search(message):
            signature_start_index = RE_AMAZON_CS_SIGNATURE_TR.search(message).start()
            return True, signature_start_index
        elif RE_AMAZON_CS_SIGNATURE_NL.search(message):
            signature_start_index = RE_AMAZON_CS_SIGNATURE_NL.search(message).start()
            return True, signature_start_index
        elif RE_AMAZON_CS_SIGNATURE_PL.search(message):
            signature_start_index = RE_AMAZON_CS_SIGNATURE_PL.search(message).start()
            return True, signature_start_index
        else:
            return False, 0    
    
    def cs_email_template_removal(self, message: str) -> Tuple[str, int]:
        '''

        :param message: input message
        :return: message that removes the Amazon CS auto signature
        '''
        message = re.sub('\xa0', ' ', message)
        RE_SIGNATURE = re.compile(r'''(?:
                                    [\s]+[-]{2,}[\s]+
                                  )
                                  ''', re.I | re.X | re.M | re.S)
        split_l = re.split(RE_SIGNATURE, message, 2) 
        
        RE_Greetings = re.compile(
            r'''(%s)[\s]*([\,!]*| [\w\s]+[\,!]*)[\s]*(?=[A-Za-z])''' % ('|'.join(Greeting_tokens)),
            re.S | re.M)
        RE_Please_reply_above_this_line = re.compile(r'''[-#]+[\s]*[\w\s]*reply above this line[\s]*[-#]+''',
                                                     re.I | re.S | re.M)

        filtered_message = []
        iffind = False
        if len(split_l) > 1:
            '''
            if can be splited by -------
            1. for message from CS central (Entry_Point == 11), extract the detail part
            '''
            split_end_index = 0
            for idx, split in enumerate(split_l):
                ifmatch, header_end_index, offset   = self.detect_amazon_cs_central_header(split)
                if ifmatch:
                    filtered_message = split[header_end_index + offset:]
                    header_end_index_all = header_end_index + split_end_index
                    iffind = True
                    return filtered_message, header_end_index_all
                else:
                    split_end_index += len(split) 
        if not iffind:
            # check for header and signature
            ifmatch_header, header_end_index, offset   = self.detect_amazon_cs_central_header(message)
            ifmatch_signature, signature_start_index   = self.detect_amazon_cs_central_signature(message)
            
            if ifmatch_header and ifmatch_signature:
                if header_end_index+offset <= signature_start_index:
                    filtered_message = message[header_end_index+offset: signature_start_index]
                    return filtered_message, header_end_index+offset
                else:
                    filtered_message = message[header_end_index+offset:]
                    return filtered_message, header_end_index+offset
            else:
                filtered_message = message
                return filtered_message, 0       

    def reply_email_line_boundary_detection(self, message: str) -> Tuple[str, int]:
        '''
        Detect the boundary of email reply
        :param message: message to be processed
        :return: the message with replies, quoatations removed
        '''
        RE_FAREWELL = re.compile(r'''(?:[.!? ]|^)(%s)''' % ('|'.join(Farewell_tokens)), re.I | re.S | re.M)
        Quotation_pattern_str_en = ['(Did this solve your problem)',
                                 '(Sent from my [\w]+)',
                                 '(Get Outlook for )',
                                 '([-]{2,}[ ]*Original Message[ ]*[-]{2,})',
                                 '([-]{2,}[ ]*Original Email[ ]*[-]{2,})',
                                 '([-]{2,}[ ]*Message[ ]*[-]{2,})',
                                 '(On .*(?=wrote:))',
                                 '([_]+\s+From: )',
                                 '(From: )',
                                 '(Sent: )',
                                 '([-]+[ ]*Forwarded message[ ]*[-]+)'
                                ]
        Quotation_pattern_str_de = ['(Von meinem [\w]+ gesendet)'
                                    '([-]{2,}[ ]*Original-Nachricht[ ]*[-]{2,})',
                                    '([-]{2,}[ ]*Ursprüngliche Daten[ ]*[-]{2,})',
                                    '(Am .*(?=schrieb\s*.*:))',
                                    '([-]{2,}\s*Originalnachricht)'
                                   ]
        Quotation_pattern_str_fr = ['(Inviato da .*(?=\s))',
                                    '([-]{2,}[ ]*Message original en langue anglaise)',
                                    '([-]{2,}[ ]*Finalizar mensaje[ ]*[-]{2,})',
                                    '(Le .*(?=wrote\s*:))',
                                    '(Le .*(?=a écrit\s*:))'
                                    
                                   ]
        Quotation_pattern_str_es = ['(Enviado desde mi [\w]+)',
                                    '(<[dirección de correo electrónico eliminada]>.*(?=escribió\s*:))',
                                    '(El .*(?=escribió\s*:))'
                                   ]  
        
        Quotation_pattern_str_it = ['([-]{2,}[ ]*Messaggio originale[ ]*[-]{2,})',
                                     '(Il .*(?=ha scritto:))'
                                   ]
        
        Quotation_pattern_str_pt = ['(Obter o Outlook para .*)',
                                    '([-]{2,}[ ]*Iniciar mensagem[ ]*[-]{2,})',
                                    '(Em .*(?=escreveu))',
                                    '([-]{2,}\s*Atualizado por:\s+)'
                                   ]
        
        Quotation_pattern_str_tr = ['(Gönderen:.* Gönderildi)',
                                    '([^.!?]+\s+cihazımdan gönderildi)',
                                    '([-]{2,}[ ]*Orijinal mesaj[ ]*[-]{2,})',
                                    '((?<=[\s.?!]) .* tarihinde .*(?=şunu yazdı:))',
                                    #'(.* tarihinde .*(?=şunu yazdı:))',
                                    '([-]{2,}\s*Gönderen:\s+)'
                                   ]
        
        Quotation_pattern_str_nl = ['([-]{2,}[ ]*Oorspronkelijk Bericht[ ]*[-]{2,})',
                                    '(Op .*(?=schreef .*:))'
                                   ]
        
        Quotation_pattern_str_pl = ["(Wysłane z .*'a)",
                                    '([-]{2,}[ ]*Ursprüngliche Nachricht[ ]*[-]{2,})',
                                    '(Wiadomość napisana przez .* (?=<\[adres e-mail usunięty\]> w dniu .*:))',
                                    '(<.*\[adres e-mail usunięty\]>\s+(?=napisał\(a\):))'
                                   ]
        
        Quotation_pattern_str_sv = ["(skrev .*<\[E-postadress borttagen\]>:)",
                                    "([_]+\s+Från:)"
                                   ]
        
        
        Quotation_pattern_str = Quotation_pattern_str_en + \
                                Quotation_pattern_str_de + \
                                Quotation_pattern_str_fr + \
                                Quotation_pattern_str_es + \
                                Quotation_pattern_str_it + \
                                Quotation_pattern_str_pt + \
                                Quotation_pattern_str_tr + \
                                Quotation_pattern_str_nl + \
                                Quotation_pattern_str_pl + \
                                Quotation_pattern_str_sv + \
                                ['(\s+[>]{1,})', '(\s+[_]{2,})']
        
        RE_BEGIN_QUOTATION = re.compile(r'''(%s)''' % ('|'.join(Quotation_pattern_str)), re.S | re.M)

        begin_quotation_match = RE_BEGIN_QUOTATION.search(message)
        farwell_match = RE_FAREWELL.search(message)
        if begin_quotation_match:
            return message[:begin_quotation_match.start()], RE_BEGIN_QUOTATION.search(message).start()
        else:
            return message, len(message)
        
    def amazon_cs_removal_step(self, message: str) -> str:
        '''
        step to remove header from Amazon, detect signature for Outlook, Gmail etc.
        '''
        
        filtered_message_1, cut_off_reply_begin_index = self.reply_email_line_boundary_detection(message)        
        filtered_message_2, cut_off_begin_index = self.cs_email_template_removal(message)
        
        # check to see which pattern shows up the first
        if cut_off_reply_begin_index < cut_off_begin_index:
            filtered_message = filtered_message_1
        elif filtered_message_2 is None:
            filtered_message = filtered_message_1
        else:
            filtered_message = filtered_message_2
            filtered_message, cut_off_reply_begin_index_2 = self.reply_email_line_boundary_detection(filtered_message)
            
        return filtered_message
    
    def process(self, message: str, **kwargs) -> str:
        if not isinstance(message, str):
            return message
        filtered_message = message.replace("\n", "").replace("\t", "")
        for func, func_name in zip(self.function_handler_list, self.function_name_list):
            filtered_message = func(filtered_message)
        
        return filtered_message        



#===============================================================================================
class AmazonMRSProcessor(Processor):
    '''
    AmazonMRSProcessor remove the email template from Merchant Return Service
    '''
    
    def __init__(self):
        self.processor_name = 'mrs_email_removal'
        self.function_handler_list = [self.remove_special_symbols, 
                                      self.mrs_email_template_removal, 
                                      self.post_process]
        self.function_name_list = ['remove_special_symbols', 
                                   'mrs_email_template_removal', 
                                   'post_process']

          
    def detect_mrs_header(self, message: str) -> Tuple[bool, int, str]:
        '''
        For message from Amazon Merchant Return Service, (Entry Point == 5), a fixed template is used for all messages. 
        The template looks like 
        
        Dear [Seller Name] and [Buyer Name],  This email is being sent to you by Amazon to notify and confirm that a return authorization has been requested for the item(s) listed below.   
        [Seller Name], please take action on this return request in the Manage Returns section of your seller account. You can also respond to the customer by replying to this e-mail.  
        [Buyer Name], the information below is confirmation of the items that you have requested to return to [Seller Name]. No additional action is required from you at this time.  
        Order ID: xxx-xxxxxxx-xxxxxxx 
        Item: xxx 
        Qty: x 
        Return reason: [Incompatible or not useful] -> relevant information  
        Request received: [DATE]   
        Sincerely,  Amazon Services
        
        
        We detect if such template is present and return the index of message that lists the Return reason part of message (i.e. listed under 'Return reason:').
        '''
        # Return Autorization
        RE_AMAZON_MRS_HEADER_EN = re.compile(r'''(^Dear |^).* and .*(:|,)[\s]+This (email|note) is being sent to you by .* to notify and confirm that a return authorization has been requested for the item(\(s\))* listed below.*(?=Return reason:\s*)''', re.M | re.S)
        RE_AMAZON_MRS_HEADER_FR = re.compile(r'''Chers .* et .*(:|,)[\s]+Cet e-mail vous est envoyé par .* afin de vous informer qu’une demande d’autorisation de retour a été soumise pour le ou les articles listés ci-dessous.*(?=Raison du retour:\s*)''', re.M | re.S)
        RE_AMAZON_MRS_HEADER_FR_3 = re.compile(r'''Chers .* et .*(:|,)[\s]+Cet e-mail vous est envoyé par .* afin de vous informer et de vous confirmer qu’une demande d’autorisation de retour a été soumise pour le ou les articles listés ci-dessous.*(?=Raison du retour.*)''', re.M | re.S)
        RE_AMAZON_MRS_HEADER_PT = re.compile(r'''Prezados .* e .*(:|,)[\s]+Este e-mail lhe foi enviado pela .* para confirmar que um pedido de autorização para devolução foi emitido para o\(s\) item(\(ns\))* abaixo.*(?=Razão para a devolução:\s*)''', re.M | re.S)
        RE_AMAZON_MRS_HEADER_DE = re.compile(r'''Guten Tag,[\s]+Amazon.* sendet Ihnen diese E-Mail zur Bestätigung, dass ein Rücksendeantrag für folgende\(n\) Artikel gestellt wurde.*(?=Rücksendegrund:\s*)''', re.M | re.S)
        RE_AMAZON_MRS_HEADER_MX = re.compile(r'''Estimados .* y .*(:|,)[\s]+Amazon os envía este correo electrónico para dejar constancia de la solicitud de autorización presentada para efectuar la devolución de los artículos indicados más adelante.*(?=Motivo de la devolución:\s*)''', re.M | re.S)
        RE_AMAZON_MRS_HEADER_IT = re.compile(r'''Gentili .* e .*(:|,)[\s]+questa e-mail vi viene inviata da Amazon quale notifica e conferma della richiesta di autorizzazione ad effettuare il reso degli articoli elencati di seguito.*(?=Motivo del reso:\s*)''', re.M | re.S)
        RE_AMAZON_MRS_HEADER_TR = re.compile(r'''Sayın .* ve .*(:|,)[\s]+Bu not, aşağıdaki ürünler için bir iade onayının istendiğine dair sizi bilgilendirmek ve bunu teyit etmek üzere Amazon tarafından gönderilmiştir.*(?=İade nedeni:\s*)''', re.M | re.S)
        RE_AMAZON_MRS_HEADER_NL = re.compile(r'''Beste .* en .*(:|,)[\s]+Amazon stuurt je deze e-mail om te bevestigen dat er autorisatie voor een retourzending is aangevraagd voor de onderstaande items.*(?=Reden voor retourneren:\s*)''', re.M | re.S)
        RE_AMAZON_MRS_HEADER_SV = re.compile(r'''Kära .* och .*(:|,)[\s]+Detta e-postmeddelande skickas till dig av Amazon för att meddela och bekräfta att ett returtillstånd har begärts för följande artiklar.*(?=Anledning till retur:\s*)''', re.M | re.S)
        RE_AMAZON_MRS_HEADER_JA = re.compile(r'''このメールは、出品者様および購入者様の両方にお送りしております。.*Amazon.co.jpよりお知らせいたします。以下の商品について返品が依頼されました。.*(?=返品理由:\s*)''', re.M | re.S)
        RE_AMAZON_MRS_HEADER_PL = re.compile(r'''Witaj .* i .*(:|,|!)[\s]+Tę wiadomość e-mail otrzymujesz od Amazon w celu powiadomienia i potwierdzenia, że wpłynęło żądanie zatwierdzenia zwrotu produktów wymienionych poniżej.*(?=Przyczyna zwrotu:\s*)''', re.M | re.S)
        
        # Return Cancellation
        RE_AMAZON_MRS_HEADER_EN_2 = re.compile(r'''Dear .* and .*, This note has been sent by Amazon to inform you and confirm that .* has cancelled the refund request for the following order.*(?=Return reason:\s*)''', re.M | re.S)
        RE_AMAZON_MRS_HEADER_FR_2 = re.compile(r'''Chers .* et .*, Cet e-mail vous est envoyé par Amazon pour vous faire part du fait que .* a annulé sa demande de retour pour la commande suivante.*(?=Raison du retour:\s*)''', re.M | re.S)
        RE_AMAZON_MRS_HEADER_TR_2 = re.compile(r'''Sayın .* ve .*,[\s]+Bu not, .* adlı alıcının aşağıdaki siparişle ilgili iade talebini iptal ettiğine dair sizi bilgilendirmek ve bunu teyit etmek üzere Amazon tarafından gönderilmiştir.*(?=İade nedeni:\s*)''', re.M | re.S)
        
        if RE_AMAZON_MRS_HEADER_EN.search(message):
            header_end_index = RE_AMAZON_MRS_HEADER_EN.search(message).end()
            return True, header_end_index, 'Return Request.'
        elif RE_AMAZON_MRS_HEADER_FR.search(message):
            header_end_index = RE_AMAZON_MRS_HEADER_FR.search(message).end()
            return True, header_end_index, 'Demande de retour.'
        elif RE_AMAZON_MRS_HEADER_FR_3.search(message):
            header_end_index = RE_AMAZON_MRS_HEADER_FR_3.search(message).end()
            return True, header_end_index, 'Demande de retour.'        
        elif RE_AMAZON_MRS_HEADER_DE.search(message):
            header_end_index = RE_AMAZON_MRS_HEADER_DE.search(message).end()
            return True, header_end_index, 'Rückgabeanfrage.'
        elif RE_AMAZON_MRS_HEADER_PT.search(message):
            header_end_index = RE_AMAZON_MRS_HEADER_PT.search(message).end()
            return True, header_end_index, 'Solicitação de Devolução.'
        elif RE_AMAZON_MRS_HEADER_MX.search(message):
            header_end_index = RE_AMAZON_MRS_HEADER_MX.search(message).end()
            return True, header_end_index, 'Solicitud de devolución.'
        elif RE_AMAZON_MRS_HEADER_IT.search(message):
            header_end_index = RE_AMAZON_MRS_HEADER_IT.search(message).end()
            return True, header_end_index, 'Richiesta di reso.'
        elif RE_AMAZON_MRS_HEADER_JA.search(message):
            header_end_index = RE_AMAZON_MRS_HEADER_JA.search(message).end()
            return True, header_end_index, '返品リクエスト。'
        elif RE_AMAZON_MRS_HEADER_TR.search(message):
            header_end_index = RE_AMAZON_MRS_HEADER_TR.search(message).end()
            return True, header_end_index, 'İade Talebi.'
        elif RE_AMAZON_MRS_HEADER_NL.search(message):
            header_end_index = RE_AMAZON_MRS_HEADER_NL.search(message).end()
            return True, header_end_index, 'Retouraanvraag.'
        elif RE_AMAZON_MRS_HEADER_SV.search(message):
            header_end_index = RE_AMAZON_MRS_HEADER_SV.search(message).end()
            return True, header_end_index, 'Returförfrågan.'
        elif RE_AMAZON_MRS_HEADER_PL.search(message):
            header_end_index = RE_AMAZON_MRS_HEADER_PL.search(message).end()
            return True, header_end_index, 'Żądanie zwrotu.'
        elif RE_AMAZON_MRS_HEADER_EN_2.search(message):
            header_end_index = RE_AMAZON_MRS_HEADER_EN_2.search(message).end()
            return True, header_end_index, 'Return Cancellation.'  
        elif RE_AMAZON_MRS_HEADER_FR_2.search(message):
            header_end_index = RE_AMAZON_MRS_HEADER_FR_2.search(message).end()
            return True, header_end_index, 'Annulation de retour.'      
        elif RE_AMAZON_MRS_HEADER_TR_2.search(message):
            header_end_index = RE_AMAZON_MRS_HEADER_TR_2.search(message).end()
            return True, header_end_index, 'İade İptali.'  
        else:
            return False, 0, ''      

    def detect_mrs_signature(self, message: str) -> Tuple[bool, int]:
        '''
        For message from Amazon Merchant Return Service, (Entry Point == 5), a fixed template is used for all messages. 
        The template looks like 
        
        Dear [Seller Name] and [Buyer Name],  This email is being sent to you by Amazon to notify and confirm that a return authorization has been requested for the item(s) listed below.   
        [Seller Name], please take action on this return request in the Manage Returns section of your seller account. You can also respond to the customer by replying to this e-mail.  
        [Buyer Name], the information below is confirmation of the items that you have requested to return to [Seller Name]. No additional action is required from you at this time.  
        Order ID: xxx-xxxxxxx-xxxxxxx 
        Item: xxx 
        Qty: x 
        Return reason: [Incompatible or not useful] -> relevant information  
        Request received: [DATE]   
        Sincerely,  Amazon Services
        
        
        We detect if such template is present and return the index of message that starts with 'Request received:.*Sincerely,.*Amazon.*'.
        
        ''' 
        RE_AMAZON_MRS_SIGNATURE_EN = re.compile(r'''Request received:.* Sincerely,\s+Amazon.*''', re.M | re.S)
        RE_AMAZON_MRS_SIGNATURE_EN_2 = re.compile(r'''Request cancelled:.*''', re.M | re.S)
        RE_AMAZON_MRS_SIGNATURE_FR = re.compile(r'''Demande reçue:.* Cordialement,\s+Amazon.*''', re.M | re.S)
        RE_AMAZON_MRS_SIGNATURE_FR_2 = re.compile(r'''Requête reçue le\s*:.* Cordialement,\s+Amazon.*''', re.M | re.S)
        RE_AMAZON_MRS_SIGNATURE_FR_3 = re.compile(r'''Demande annulée:.*''', re.M | re.S)
        RE_AMAZON_MRS_SIGNATURE_PT = re.compile(r'''Pedido recebido:.* Atenciosamente,\s+Amazon.*''', re.M | re.S)        
        RE_AMAZON_MRS_SIGNATURE_DE = re.compile(r'''Eingang des Rücksendeantrags:.* Freundliche Grüße.*Amazon.*''', re.M | re.S)
        RE_AMAZON_MRS_SIGNATURE_ES = re.compile(r'''Solicitud recibida el.* Atentamente,\s+Amazon.*''', re.M | re.S)
        RE_AMAZON_MRS_SIGNATURE_IT = re.compile(r'''Richiesta ricevuta il:.* Distinti Saluti,\s+Amazon.*''', re.M | re.S)
        RE_AMAZON_MRS_SIGNATURE_TR = re.compile(r'''Alınan talep.* Saygılarımızla.*Amazon.*''', re.M | re.S)
        RE_AMAZON_MRS_SIGNATURE_TR_2 = re.compile(r'''İptal edilen talep:.*''', re.M | re.S)
        RE_AMAZON_MRS_SIGNATURE_NL = re.compile('''Verzoek ontvangen:.* Met vriendelijke groet,\s+Amazon.*''', re.M | re.S)
        RE_AMAZON_MRS_SIGNATURE_SV = re.compile(r'''Begäran mottagen:.* Med vänliga hälsningar,\s+Amazon.*''', re.M | re.S)
        RE_AMAZON_MRS_SIGNATURE_JA = re.compile(r'''返品リクエストの送信日:.*今後ともAmazon.co.jpをよろしくお願いいたします。.*''', re.M | re.S)
        RE_AMAZON_MRS_SIGNATURE_PL = re.compile(r'''返品リクエストの送信日:.*今後ともAmazon.co.jpをよろしくお願いいたします。.*''', re.M | re.S)
       

        if RE_AMAZON_MRS_SIGNATURE_EN.search(message):
            signature_start_index = RE_AMAZON_MRS_SIGNATURE_EN.search(message).start()
            return True, signature_start_index
        elif RE_AMAZON_MRS_SIGNATURE_EN_2.search(message):
            signature_start_index = RE_AMAZON_MRS_SIGNATURE_EN_2.search(message).start()
            return True, signature_start_index
        elif RE_AMAZON_MRS_SIGNATURE_FR.search(message):
            signature_start_index = RE_AMAZON_MRS_SIGNATURE_FR.search(message).start()
            return True, signature_start_index
        elif RE_AMAZON_MRS_SIGNATURE_FR_2.search(message):
            signature_start_index = RE_AMAZON_MRS_SIGNATURE_FR_2.search(message).start()
            return True, signature_start_index
        elif RE_AMAZON_MRS_SIGNATURE_FR_3.search(message):
            signature_start_index = RE_AMAZON_MRS_SIGNATURE_FR_3.search(message).start()
            return True, signature_start_index
        elif RE_AMAZON_MRS_SIGNATURE_DE.search(message):
            signature_start_index = RE_AMAZON_MRS_SIGNATURE_DE.search(message).start()
            return True, signature_start_index   
        elif RE_AMAZON_MRS_SIGNATURE_PT.search(message):
            signature_start_index = RE_AMAZON_MRS_SIGNATURE_PT.search(message).start()
            return True, signature_start_index   
        elif RE_AMAZON_MRS_SIGNATURE_ES.search(message):
            signature_start_index = RE_AMAZON_MRS_SIGNATURE_ES.search(message).start()
            return True, signature_start_index 
        elif RE_AMAZON_MRS_SIGNATURE_IT.search(message):
            signature_start_index = RE_AMAZON_MRS_SIGNATURE_IT.search(message).start()
            return True, signature_start_index 
        elif RE_AMAZON_MRS_SIGNATURE_JA.search(message):
            signature_start_index = RE_AMAZON_MRS_SIGNATURE_JA.search(message).start()
            return True, signature_start_index
        elif RE_AMAZON_MRS_SIGNATURE_TR_2.search(message):
            signature_start_index = RE_AMAZON_MRS_SIGNATURE_TR_2.search(message).start()
            return True, signature_start_index
        elif RE_AMAZON_MRS_SIGNATURE_NL.search(message):
            signature_start_index = RE_AMAZON_MRS_SIGNATURE_NL.search(message).start()
            return True, signature_start_index        
        elif RE_AMAZON_MRS_SIGNATURE_SV.search(message):
            signature_start_index = RE_AMAZON_MRS_SIGNATURE_SV.search(message).start()
            return True, signature_start_index
        else:
            return False, 0    
        
    def mrs_email_template_removal(self, message: str) -> str:
        '''
        Detect and remove Amazon blurb
        :param message: message to be processed
        :return: the message with Amazon Return center auto signature removed
        '''        
        # check for header and signature
        ifmatch_header, header_end_index, header_type   = self.detect_mrs_header(message)
        ifmatch_signature, signature_start_index   = self.detect_mrs_signature(message)
        
        if ifmatch_header and ifmatch_signature:
            if header_end_index <= signature_start_index:
                filtered_message = message[header_end_index: signature_start_index]
                filtered_message = re.sub(r'(?<=[^.!?])$', '.', filtered_message)
                filtered_message = re.sub(r'[.!?,:;_\s]+(?=[.!?])', '', filtered_message)
                return filtered_message
            else:
                filtered_message = message[:signature_start_index]
                filtered_message = re.sub(r'(?<=[^.!?])$', '.', filtered_message)
                filtered_message = re.sub(r'[.!?,:;_\s]+(?=[.!?])', '', filtered_message)
                return filtered_message
        else:
            return message
    
    def post_process(self, message: str) -> str:
        # remove removed part
        re_remove_removed_part = r"\[.*(removed|entfernt|supprimée|eliminado|eliminato|eliminada|removido|usunięty|verwijderd|borttagen)\]"
        filtered_message = re.sub(re_remove_removed_part, '', message)
        # remove trailing quotations symbol'>'
        filtered_message = re.sub(r"([\s>_]+$)", '', filtered_message)
        return filtered_message
    
    
    def process(self, message: str, **kwargs) -> str:
        if not isinstance(message, str):
            return message
        filtered_message = message.replace("\n", "").replace("\t", "")
        for func, func_name in zip(self.function_handler_list, self.function_name_list):
            filtered_message = func(filtered_message)
        return filtered_message    
    
    
    
#====================================================    
class AmazonHQRSProcessor(Processor):
    '''
    Remove Amazon email template from High Quality Review Socilitation, 3rd Party API
    '''
    
    def __init__(self):
        self.processor_name = 'hqrs_email_removal'
        self.function_handler_list = [self.remove_special_symbols, 
                                      self.hqrs_email_template_removal, 
                                      self.sentence_norm
                                     ]
        self.function_name_list = ['remove_special_symbols', 
                                   'hqrs_email_template_removal', 
                                   'sentence_norm'
                                  ]

        
    def detect_hqrs_header(self, message: str) -> Tuple[bool, int]:
        '''
        For message from HQRS from 3P Developers via the API, (Entry Point == 20), a fixed template is used for all messages. 
        The template looks like
        
        Hi [BUYER]:  Thank you for shopping with us. Amazon seller [SELLER] has encountered an unexpected problem with completing your order.  
        --- Message from seller [SELLER]:  
        Dear customer,  Thanks so much for choosing our product.   
        Order ID xxx-xxxxxxx-xxxxxxx ASIN xxxxxxxx Wish the parcel has been delivered to you by now.   
        Please feel free to contact us if any issues and defects on the product. 
        Before selecting a return, you can reply  to this email and describe the problem in detail. 
        We will consult with supplier for solution and try our best to offer the most convenient service for you.   
        The shipment is fulfilled by Amazon logistics. If you haven't received the package, please don't worry, there may be a little delay in transit. 
        You can stay tuned for the update of the tracking information. It is also available to consult  the local carrier or Amazon customer service directly to check.  
        Hope you have a happy shopping.  Best regards  
        ---
        
        We detect if such template is present and return the index of original message
        between '--- Message from seller [SELLER]:' and '---'
        
        '''
        message = re.sub('\xa0', ' ', message)
        
        RE_AMAZON_3P_EN = re.compile(r'''Message from seller [\w\d\-\s]+:[\s]+(?=[\w])''', re.S | re.M)
        RE_AMAZON_3P_FR = re.compile(r'''Message du vendeur [\w\d\-\s]+:[\s]+(?=[\w])''', re.S | re.M)
        RE_AMAZON_3P_DE = re.compile(r'''Nachricht vom Verkäufer [\w\d\-\s]+:[\s]+(?=[\w])''', re.S | re.M)
        RE_AMAZON_3P_PT = re.compile(r'''Mensagem do vendedor [\w\d\-\s]+:[\s]+(?=[\w])''', re.S | re.M)
        RE_AMAZON_3P_ES = re.compile(r'''Mensaje del vendedor [\w\d\-\s]+:[\s]+(?=[\w])''', re.S | re.M)
        RE_AMAZON_3P_IT = re.compile(r'''Messaggio dal venditore [\w\d\-\s]+:[\s]+(?=[\w])''', re.S | re.M)
        RE_AMAZON_3P_NL = re.compile(r'''Bericht van verkoper [\w\d\-\s]+:[\s]+(?=[\w])''', re.S | re.M)
        RE_AMAZON_3P_PL = re.compile(r'''Wiadomość od sprzedawcy [\w\d\-\s]+:[\s]+(?=[\w])''', re.S | re.M)
        RE_AMAZON_3P_JA = re.compile(r'''出品者.*からのメッセージ：\s+''', re.S | re.M)
        
        if RE_AMAZON_3P_EN.search(message):
            header_end_index = RE_AMAZON_3P_EN.search(message).end()
            return True, header_end_index
        elif RE_AMAZON_3P_FR.search(message):
            header_end_index = RE_AMAZON_3P_FR.search(message).end()
            return True, header_end_index
        elif RE_AMAZON_3P_DE.search(message):
            header_end_index = RE_AMAZON_3P_DE.search(message).end()
            return True, header_end_index
        elif RE_AMAZON_3P_PT.search(message):
            header_end_index = RE_AMAZON_3P_PT.search(message).end()
            return True, header_end_index
        elif RE_AMAZON_3P_ES.search(message):
            header_end_index = RE_AMAZON_3P_ES.search(message).end()
            return True, header_end_index
        elif RE_AMAZON_3P_IT.search(message):
            header_end_index = RE_AMAZON_3P_IT.search(message).end()
            return True, header_end_index
        elif RE_AMAZON_3P_NL.search(message):
            header_end_index = RE_AMAZON_3P_NL.search(message).end()
            return True, header_end_index
        elif RE_AMAZON_3P_PL.search(message):
            header_end_index = RE_AMAZON_3P_PL.search(message).end()
            return True, header_end_index       
        elif RE_AMAZON_3P_PL.search(message):
            header_end_index = RE_AMAZON_3P_PL.search(message).end()
            return True, header_end_index  
        else:
            return False, 0   
  

    def hqrs_email_template_removal(self, message: str) -> str:
        '''
        Detect and remove Amazon blurb
        :param message: message to be processed
        :return: the message with Amazon Return center auto signature removed
        '''        
        message = re.sub('\xa0', ' ', message)
        
        # check for header and signature
        ifmatch_header, header_end_index   = self.detect_hqrs_header(message)
        
        RE_AMZON_3P_END = re.compile(r'''([\s]+[-]{2,}$)''', re.I | re.X | re.M | re.S)
        ifmatch_signature = True if RE_AMZON_3P_END.search(message) else False
        if ifmatch_signature:
            signature_start_index = RE_AMZON_3P_END.search(message).start()
        
        if ifmatch_header:
            if ifmatch_signature:
                filtered_message = message[header_end_index: signature_start_index]
                filtered_message = re.sub(r'(?<=[^.!?])$', '.', filtered_message)
                filtered_message = re.sub(r'[.!?,:;_\s]+(?=[.!?])', '', filtered_message)
                return filtered_message 
            else:
                filtered_message = message[header_end_index:]
                # remove trailing quotations symbol'>'
                filtered_message = re.sub(r"([\s>_]+$)", '', filtered_message)
                filtered_message = re.sub(r'(?<=[^.!?])$', '.', filtered_message)
                filtered_message = re.sub(r'[.!?,:;_\s]+(?=[.!?])', '', filtered_message)
                return filtered_message 
        else:
            return message
        
           
    def process(self, message: str, **kwargs) -> str:
        if not isinstance(message, str):
            return message
        filtered_message = message.replace("\n", "").replace("\t", "")
        for func, func_name in zip(self.function_handler_list, self.function_name_list):
            filtered_message = func(filtered_message)
        
        return filtered_message                 

    
    
#====================================================
class EmailProcessor(Processor):
    '''
    Remove quotations of previous emails from Gmail, Hotmail etc.
    '''
    
    def __init__(self):
        self.processor_name = 'normal_email'
        self.function_handler_list = [self.remove_special_symbols, 
                                      self.email_reply_removal, 
                                      self.sentence_norm
                                     ]
        self.function_name_list = ['remove_special_symbols', 
                                   'hqrs_email_template_removal', 
                                   'sentence_norm'
                                  ]
    
    
    def reply_email_line_boundary_detection(self, message: str) -> Tuple[str, int]:
        '''
        Detect the boundary of email reply
        :param message: message to be processed
        :return: the message with replies, quoatations removed
        '''
        RE_FAREWELL = re.compile(r'''(?:[.!? ]|^)(%s)''' % ('|'.join(Farewell_tokens)), re.I | re.S | re.M)
        Quotation_pattern_str_en = ['(Did this solve your problem)',
                                 '(Sent from my [\w]+)',
                                 '(Get Outlook for )',
                                 '(Download Outlook for iOS<[\_]+)',
                                 '([-]{2,}[ ]*Original Message[ ]*[-]{2,})',
                                 '([-]{2,}[ ]*Original Email[ ]*[-]{2,})',
                                 '([-]{2,}[ ]*Message[ ]*[-]{2,})',
                                 '(On .*(?=wrote:))',
                                 '([_]+\s+From: )',
                                 '(From: )',
                                 '(Sent: )',
                                 '([-]+[ ]*Forwarded message[ ]*[-]+)'
                                ]
        Quotation_pattern_str_de = ['(Von meinem [\w]+ gesendet)'
                                    '([-]{2,}[ ]*Original-Nachricht[ ]*[-]{2,})',
                                    '([-]{2,}[ ]*Ursprüngliche Daten[ ]*[-]{2,})',
                                    '(Am .*(?=schrieb\s*.*:))',
                                    '([-]{2,}\s*Originalnachricht)'
                                   ]
        Quotation_pattern_str_fr = ['(Inviato da .*(?=\s))',
                                    '([-]{2,}[ ]*Message original en langue anglaise)',
                                    '([-]{2,}[ ]*Finalizar mensaje[ ]*[-]{2,})',
                                    '(Le .*(?=wrote\s*:))',
                                    '(Le .*(?=a écrit\s*:))'
                                    
                                   ]
        Quotation_pattern_str_es = ['(Enviado desde mi [\w]+)',
                                    '(<[dirección de correo electrónico eliminada]>.*(?=escribió\s*:))',
                                    '(El .*(?=escribió\s*:))'
                                   ]  
        
        Quotation_pattern_str_it = ['([-]{2,}[ ]*Messaggio originale[ ]*[-]{2,})',
                                     '(Il .*(?=ha scritto:))'
                                   ]
        
        Quotation_pattern_str_pt = ['(Obter o Outlook para .*)',
                                    '([-]{2,}[ ]*Iniciar mensagem[ ]*[-]{2,})',
                                    '(Em .*(?=escreveu))',
                                    '([-]{2,}\s*Atualizado por:\s+)'
                                   ]
        
        Quotation_pattern_str_tr = ['(Gönderen:.* Gönderildi)',
                                    '([^.!?]+\s+cihazımdan gönderildi)',
                                    '([-]{2,}[ ]*Orijinal mesaj[ ]*[-]{2,})',
                                    '(tarihinde .*(?=şunu yazdı:))',
                                    #'(.* tarihinde .*(?=şunu yazdı:))',
                                    '([-]{2,}\s*Gönderen:\s+)'
                                   ]
        
        Quotation_pattern_str_nl = ['([-]{2,}[ ]*Oorspronkelijk Bericht[ ]*[-]{2,})',
                                    '(Op .*(?=schreef .*:))'
                                   ]
        
        Quotation_pattern_str_pl = ["(Wysłane z .*'a)",
                                    '([-]{2,}[ ]*Ursprüngliche Nachricht[ ]*[-]{2,})',
                                    '(Wiadomość napisana przez .* (?=<\[adres e-mail usunięty\]> w dniu .*:))',
                                    '(<.*\[adres e-mail usunięty\]>\s+(?=napisał\(a\):))'
                                   ]
        
        Quotation_pattern_str_sv = ["(skrev .*<\[E-postadress borttagen\]>:)",
                                    "([_]+\s+Från:)"
                                   ]
        
        
        Quotation_pattern_str = Quotation_pattern_str_en + \
                                Quotation_pattern_str_de + \
                                Quotation_pattern_str_fr + \
                                Quotation_pattern_str_es + \
                                Quotation_pattern_str_it + \
                                Quotation_pattern_str_pt + \
                                Quotation_pattern_str_tr + \
                                Quotation_pattern_str_nl + \
                                Quotation_pattern_str_pl + \
                                Quotation_pattern_str_sv + \
                                ['(\s+[>]{1,})', '(\s+[_]{2,})']
        
        RE_BEGIN_QUOTATION = re.compile(r'''(%s)''' % ('|'.join(Quotation_pattern_str)), re.S | re.M)

        begin_quotation_match = RE_BEGIN_QUOTATION.search(message)
        farwell_match = RE_FAREWELL.search(message)
        if begin_quotation_match:
            return message[:begin_quotation_match.start()], RE_BEGIN_QUOTATION.search(message).start()
        else:
            return message, len(message)

        
    def detect_reply_above_this_line_header(self, message: str) -> Tuple[bool, int]:
        RE_REPLY_ABOVE_THIS_LINE_EN = re.compile(r'''[-#]+[\s]*Please reply above this line[\s]*[-#]+[\s]*''',
                                                     re.I | re.S | re.M)
        
        RE_REPLY_ABOVE_THIS_LINE_DE = re.compile(r'''[-#]+[\s]*Lütfen yanıtınızı bu satırın üzerine yazın[\s]*[-#]+[\s]*''',
                                                     re.I | re.S | re.M)    
        
        if RE_REPLY_ABOVE_THIS_LINE_EN.search(message):
            header_end_index = RE_REPLY_ABOVE_THIS_LINE_EN.search(message).end()
            return True, header_end_index
        elif RE_REPLY_ABOVE_THIS_LINE_DE.search(message):
            header_end_index = RE_REPLY_ABOVE_THIS_LINE_DE.search(message).end()
            return True, header_end_index
        else:
            return False, len(message)
               
        
    def email_reply_removal(self, message: str) -> str:
        '''
           detect reply line pattern Outlook, Gmail etc.
        '''
        ifmatch, reply_above_this_line_header_end_index = self.detect_reply_above_this_line_header(message)  
        filtered_message, cut_off_reply_begin_index = self.reply_email_line_boundary_detection(message)        
        
        if ifmatch:
            if reply_above_this_line_header_end_index < cut_off_reply_begin_index:
                filtered_message = message[reply_above_this_line_header_end_index:cut_off_reply_begin_index]
        
        # check to see which pattern shows up the first
        #if cut_off_reply_begin_index == 0:
        #    filtered_message = filtered_message_2
        #    filtered_message, cut_off_reply_begin_index_2 = self.reply_email_line_boundary_detection(filtered_message)
            
        return filtered_message  
    
    
    def process(self, message: str, **kwargs) -> str:
        if not isinstance(message, str):
            return message
        filtered_message = message.replace("\n", "").replace("\t", "")
        for func, func_name in zip(self.function_handler_list, self.function_name_list):
            filtered_message = func(filtered_message)
        
        return filtered_message 
  
    
#====================================================================================================    
class LabelProcessor(Processor):
    '''
    Remove quotations of previous emails from Gmail, Hotmail etc.
    '''
    label_encoder: sklearn.preprocessing._label.LabelEncoder
    label_encoder_dict: dict
    
    def __init__(self, label_encoder: Optional[sklearn.preprocessing._label.LabelEncoder] = None,
                 label_encoder_dict: Dict[Union[str, int], int] = None
                ):
        self.processor_name = 'label_encoding_process'
        if not label_encoder or not isinstance(label_encoder, sklearn.preprocessing._label.LabelEncoder):
            self.label_encoder = LabelEncoder()
        else:
            self.label_encoder = label_encoder
            
        try:
            self.label_encoder_dict = dict(zip(self.label_encoder.classes_, 
                                self.label_encoder.transform(self.label_encoder.classes_).tolist()))
        except AttributeError:
            if label_encoder_dict:
                self.label_encoder_dict = label_encoder_dict
            else:
                self.label_encoder_dict = dict()
        
        self.function_handler_list = [
                                      self.label_transform, 
                                     ]
        self.function_name_list = [
                                   'label_transform', 
                                  ]
        
    def label_encoder_fit(self, label_df: Union[list, pd.core.series.Series]) -> None:
        if not isinstance(label_df, list) and \
            not isinstance(label_df, pd.core.series.Series) and \
            not isinstance(label_df, pd.core.frame.DataFrame):
            raise TypeError('Input must be list or pandas.core.series.Series or pandas.core.frame.DataFrame')
        self.label_encoder.fit(label_df)
        self.label_encoder_dict = dict(zip(self.label_encoder.classes_, 
                                self.label_encoder.transform(self.label_encoder.classes_).tolist()))

        
    def label_transform(self, label: Union[str, int, list, pd.core.series.Series]) -> Union[int, List[int]]:
        if not isinstance(label, list) and \
            not isinstance(label, pd.core.series.Series) and \
            not isinstance(label, pd.core.frame.DataFrame):
                
            if isinstance(label, str) or isinstance(label, int):
                label = [label]
                try:
                    result = self.label_encoder.transform(label)
                except ValueError:
                    result = [-1]
                return result[0]
            else:
                raise TypeError('Input must be list or pandas.core.series.Series')
        else:
            try:
                transformed_label = self.label_encoder.transform(label)
            except ValueError:
                transformed_label = -1*np.ones(label.shape)
            return transformed_label
    
    def process(self, label: Union[str, int,  list, pd.core.series.Series], **kwargs) -> Union[int, List[int]]:
        if not self.label_encoder_dict or len(self.label_encoder_dict) == 0:
            return label
        else:
            return self.label_transform(label)
    

#====================================================================================================    
class OrdinalProcessor(Processor):
    '''
    Remove quotations of previous emails from Gmail, Hotmail etc.
    '''    
    def __init__(self, encoder: Optional[sklearn.preprocessing._encoders.OrdinalEncoder] = None,
                 encoder_dict: Dict[Union[str, int], int] = None
                ):
        self.processor_name = 'categorical_encoder'
        if not encoder or not isinstance(encoder, sklearn.preprocessing._encoders.OrdinalEncoder):
            self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        else:
            self.encoder = encoder
            
        try:
            self.encoder_dict = dict(zip(list(self.encoder.categories_[0]), 
                                              self.encoder.transform(self.encoder.categories_[0].reshape(-1, 1)).reshape(1, -1).tolist()[0]
                                    ))
        except AttributeError:
            if encoder_dict:
                self.encoder_dict = encoder_dict
            else:
                self.encoder_dict = dict()
        
        self.function_handler_list = [
                                      self.feature_transform, 
                                     ]
        self.function_name_list = [
                                   'feature_transform', 
                                  ]
        
    def encoder_fit(self, df: Union[list, pd.core.series.Series]) -> None:
        df_array = None
        if isinstance(df, list):
            df_array = np.array(df).reshape(-1, 1)
        elif isinstance(df, pd.core.series.Series):
            df_array = df.to_numpy().reshape(-1, 1)
        elif isinstance(df, pd.core.frame.DataFrame):
            df_array = df.to_numpy()
        else:
            raise TypeError('Input must be list or pandas.core.series.Series or pandas.core.frame.DataFrame')
            
        self.encoder.fit(df_array)
        self.encoder_dict = dict(zip(list(self.encoder.categories_[0]), 
                                          self.encoder.transform(self.encoder.categories_[0].reshape(-1, 1)).reshape(1, -1).tolist()[0]
                                ))

        
    def feature_transform(self, 
                          categorical: Union[str, int, list, pd.core.series.Series, pd.core.frame.DataFrame]) -> Union[int, List[int]]:
        if isinstance(categorical, str) or isinstance(categorical, int):
            categorical_array = [[categorical]]
        elif isinstance(categorical, list):
            categorical_array = np.array(categorical).reshape(-1, 1)
        elif isinstance(categorical, pd.core.series.Series):
            categorical_array = categorical.to_numpy().reshape(-1, 1)
        elif isinstance(categorical, pd.core.frame.DataFrame):
            categorical_array = categorical.to_numpy()
        else:
            raise TypeError('Input must be int, str, list, pandas.core.series.Series or pandas.core.frame.DataFrame')
            
        transformed_label = self.encoder.transform(categorical_array)
        if isinstance(categorical, str) or isinstance(categorical, int):
            transformed_label = transformed_label[0][0]
        elif isinstance(categorical, list) or isinstance(categorical, pd.core.series.Series):
            transformed_label = transformed_label.reshape(1, -1)[0]
            
        return transformed_label
    
    def process(self, 
                categorical: Union[str, int,  list, pd.core.series.Series, pd.core.frame.DataFrame], 
                **kwargs) -> Union[int, List[int]]:
        if not self.encoder_dict or len(self.encoder_dict) == 0:
            return categorical
        else:
            return self.feature_transform(categorical)
        
    

#=======================

# --- Dummy Tokenizer for Unit Testing (can also be used by others) ---
class DummyTokenizer:
    """A dummy tokenizer that splits text by whitespace."""
    def encode(self, text, add_special_tokens=False):
        return text.split()
    
    def __call__(self, text, add_special_tokens=True, max_length=None, truncation=True, padding="longest", return_attention_mask=True):
        tokens = text.split()
        return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}