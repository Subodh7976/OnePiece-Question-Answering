# OnePiece Question Answering

OnePiece Question Answering is a tool to answer queries related to OnePiece Anime. OnePiece is an anime which consists 1000+ chapters, so it becomes hard to search for a single query you have in mind. With this tool you can ask any query related to the OnePiece anime (works better if it involves major keyword). This project scraped over 12000 articles from [OnePiece Fandom]() which contains details related to every character, chapter and every incidents. This project utilizes the Roberta-base model as its base Document Reader to answer query from a context and BM25 as document retriever. The articats folder is not included (because of its big size) and the data craping and model training can take time or you can download articats contents directly from [here]().

## Getting Started

To get started with using this tool, you might want to clone or download the repository.

```bash
git clone https://github.com/Subodh7300/OnePiece-Question-Answering
cd OnePiece-Question-Answering
```

## Dependencies

You might have to install multiple dependencies, so just run the command to directly install all the dependecies.
```bash
pip install -r requirements.txt
```

## Usage

Now once installed, launch it by running the app.py python file, and the flask server will start on localhost.
```bash
python app.py
```

You can visit http://127.0.0.1:5000/ or http://localhost:5000/ to access the web interface.

## Contributing

For any changes, please open an issue 
to discuss what you would like to change.

As I am new to this, any contribution is welcome.

## Want to contact me?
* [Telegram](https://t.me/subodh79)
* [LinkedIn](https://www.linkedin.com/in/subodh-uniyal-655328230)
* [Instagram](https://www.instagram.com/subodh_7300/)
* [Email](s.subodh7976@gmail.com)