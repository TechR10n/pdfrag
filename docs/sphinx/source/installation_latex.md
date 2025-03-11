% Auto-generated LaTeX document
\input{preamble}

\begin{document}

\section{Installation}

This section covers how to install and set up the PDF RAG System.

\subsection{Requirements}

\begin{itemize}
\item Python 3.8 or higher
\item pip (Python package manager)
\item Virtual environment (recommended)
\end{itemize}

\subsection{Installation Steps}

\begin{itemize}
\item Clone the repository:
\end{itemize}
   \begin{lstlisting}[language=bash]
git clone https://github.com/yourusername/pdfrag.git
   cd pdfrag
\end{lstlisting}

\begin{itemize}
\item Create and activate a virtual environment:
\end{itemize}
   \begin{lstlisting}[language=bash]
python -m venv .venv
   source .venv/bin/activate  \# On Windows: .venv\Scripts\activate
\end{lstlisting}

\begin{itemize}
\item Install the required packages:
\end{itemize}
   \begin{lstlisting}[language=bash]
pip install -r requirements.txt
\end{lstlisting}

\begin{itemize}
\item Set up the configuration:
\end{itemize}
   \begin{lstlisting}[language=bash]
cp app/config/config.example.py app/config/config.py
   \# Edit app/config/config.py with your settings
\end{lstlisting}

\begin{itemize}
\item Run the application:
\end{itemize}
   \begin{lstlisting}[language=bash]
python app/main.py
\end{lstlisting}

The application should now be running at http://localhost:5000.


\end{document}
