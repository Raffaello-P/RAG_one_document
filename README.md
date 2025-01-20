# RAG_one_document
Demo di un sistema RAG in grado di fornire risposte riguardanti un determinato documento. Demo per caso d'uso in cui un PM ha bisogno di reperire informazioni riguardanti il progetto basandosi sulla documentazione tecnica.

# LLM Query Tool

Un semplice strumento per interrogare un LLM tramite Ollama.

## Come usare
1. Installa le dipendenze: `pip install -r requirements.txt`
2. Inserisci il documento tecnico nella cartella "documents"
3. Configura i parametri in `config/config.json`. In questo file è possibile anche configurare/cambiare modello.
4. Nel main è presente una variabile "pdf_path" in cui bisogna inserire il nome del documento (è presente il commento: inserire nome del documento).
5. Avvia lo script: `python -m src.main`.

## Da ricordare per l'implementazione
1. Modelli da 2 miliardi di parametri hanno limiti tipici di 2048 token o 4096 token.
2. Questo limite include contesto + query + risposta prevista.

## Info codice
1. nltk.download('punkt') Scarica il pacchetto per il tokenizing. Il Punkt tokenizer è un modulo pre-addestrato per la segmentazione del testo in:
    Frasi: Identifica i confini delle frasi (ad esempio, dove finisce una frase e inizia un'altra).
    Parole: Divide una frase in singole parole o token.

    Il modello Punkt è basato su un'analisi statistica del linguaggio naturale ed è progettato per:
        a. Riconoscere i confini delle frasi anche in situazioni ambigue (es. abbreviazioni come "Dr.", "etc.", che non devono essere interpretate come fine frase).
        b. Gestire vari linguaggi, come inglese, italiano, tedesco, ecc.

    Comandi:
        a. nltk.sent_tokenize(text): Per dividere un testo in frasi.
        b. nltk.word_tokenize(text): Per dividere una frase in parole/token.
    
    NOTA BENE: forse è da modificare con nltk.download('punkt_tab')
 
