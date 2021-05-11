# NER-label-the-Maori-language-via-BiLSTM-CRF
I select some open source experience and label the Maori Language by using BiLSTM-CRF
NER_the Maori
***************
This project uses named entity recognition technology to name text entities in Maori language. 
Through this project, we can separate entities in Maori texts and output them.
The dataset comes from 1450 Maori language data and 217646 English language data.

Features
***************
 - Be awesome

 - Can separate the less resources language particular in the te reo Maori

 - Use the most popular deep learning algorithms, BiLSTM-CRF


Environment
***************
-- Hardware--
CPU: Intel Core i7-8565U
RAM: 8.00GB

--Software--
 - PyCharm Community Edition
 - Python Packages & Version
   backports.weakref==1.0rc1 
   bleach==1.5.0 
   boto==2.48.0 
   bz2file==0.98 
   certifi==2017.11.5 
   chardet==3.0.4 
   enum34==1.1.6 
   gensim==3.1.0 
   h5py==2.7.1 
   html5lib==0.9999999 
   idna==2.6 
   Keras==2.2.0 
   m2r==0.1.12 
   Markdown==2.6.9 
   numpy==1.13.3 
   protobuf==3.5.1 
   python-dateutil==2.6.0 
   pytz==2017.2 
   PyYAML==4.2b1 
   requests==2.21.0 
   scikit-learn==0.19.1 
   scipy==1.0.0 
   seqeval==0.0.3 
   six==1.11.0 
   smart-open==1.5.3 
   tensorboard==1.8.0 
   tensorflow==1.8.0 
   Theano==0.9.0 
   urllib3==1.24.1 
   Werkzeug==0.15.3 
   allennlp==0.7.1 


Installation & Files Explanation
***************
 - Install NER_the Maori project by running the following three files:
     -training_example.py
       This file is about the training corpus. It converts the corpus into digital vectors, and puts these digital vectors into the model for training.
     -models.py
       This file stores the relevant model parameters. The key parameters of the model can be modified in this file.
     -tagger_example.py
       This file implements the NER function. In this file, for the complete Maori sentence, we segment, and finally get the entity in the complete Maori sentence.
     -NER-Output.txt
       This file saves the Maori entities results that come from the project. 
 - train.txt
    Saving the text corpus.
 - valid.txt 
    Saving the text corpus.
 - params.json & preprocessor.pkl & weights.h5
    Saving parameters information, can be called by models.

Operation Instruction Manual
***************
-Data Loading
  def main(args):
        print('Loading dataset...')
        x_train, y_train = load_data_and_labels(args.train_data)
        x_valid, y_valid = load_data_and_labels(args.valid_data)

       print('Transforming datasets...')
       p = IndexTransformer(use_char=args.no_char_feature)
       p.fit(x_train, y_train)

-Build Word Embedding
  def build(self):
        word_ids = Input(batch_shape=(None, None), dtype='int32', name='word_input')
        if self._embedding is None:
              word_embeddings = Embedding(input_dim=self._word_vocab_size,
                                                                   output_dim=self._word_embedding_dim,
                                                                   mask_zero=True,
                                                                   name='word_embedding')(word_ids)

-Build BiLSTM-CRF Model
  print('Building a model.')
  model = BiLSTMCRF(char_embedding_dim=args.char_emb_size,
                                       word_embedding_dim=args.word_emb_size,
                                       char_lstm_size=args.char_lstm_units, 
                                       word_lstm_size=args.word_lstm_units,
                                       char_vocab_size=p.char_vocab_size,
                                       word_vocab_size=p.word_vocab_size,
                                       num_labels=p.label_size,
                                       dropout=args.dropout,
                                       use_char=args.no_char_feature,
                                       use_crf=args.no_use_crf)
  model, loss = model.build()
  model.compile(loss=loss, optimizer='adam')

-Parameters functions
  if __name__ == '__main__':
    DATA_DIR = os.path.join(os.path.dirname(__file__), '../data/conll2003/en/ner')
    parser = argparse.ArgumentParser(description='Training a model')
    parser.add_argument('--train_data', default=os.path.join(DATA_DIR, 'train.txt'), help='training data')
    parser.add_argument('--valid_data', default=os.path.join(DATA_DIR, 'valid.txt'), help='validation data')
    parser.add_argument('--weights_file', default='weights.h5', help='weights file')
    parser.add_argument('--params_file', default='params.json', help='parameter file')
    # parser.add_argument('--preprocessor_file', default='preprocessor.json', help='preprocessor file')
    parser.add_argument('--preprocessor_file', default='preprocessor.pkl', help='preprocessor file')
    # Training parameters
    parser.add_argument('--loss', default='categorical_crossentropy', help='loss')
    parser.add_argument('--optimizer', default='adam', help='optimizer')
    parser.add_argument('--max_epoch', type=int, default=5, help='max epoch')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--checkpoint_path', default=None, help='checkpoint path')
    parser.add_argument('--log_dir', default=None, help='log directory')
    parser.add_argument('--early_stopping', action='store_true', help='early stopping')
    # Model parameters
    parser.add_argument('--char_emb_size', type=int, default=25, help='character embedding size')
    parser.add_argument('--word_emb_size', type=int, default=100, help='word embedding size')
    parser.add_argument('--char_lstm_units', type=int, default=25, help='num of character lstm units')
    parser.add_argument('--word_lstm_units', type=int, default=100, help='num of word lstm units')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--no_char_feature', action='store_false', help='use char feature')
    parser.add_argument('--no_use_crf', action='store_false', help='use crf layer')

-Training
  model = anago.Sequence()
  model.fit(x_train, y_train, epochs=15)

-Output Function
  def main(args):
        print('Loading objects...')
        model = models.load_model(args.weights_file, args.params_file)
        it = IndexTransformer.load(args.preprocessor_file)
        tagger = Tagger(model, preprocessor=it)

       print('Tagging a sentence...')
       res = tagger.analyze(args.sent)
       print(res)
       while True:
              print('Input a Sentence:')
              input_sentence = input()
              res = tagger.analyze(input_sentence)
              print(res)
              print('\n')

Developers & Users
***************
Jian Lu
Email Address: deer690535256@gmail.com


Support
************
  If you are having issues, please let us know. 
  We have an email that you can connect with us.: deer690535256@gmail.com


License
************
  The project is licensed under the GPL license.


References
************
  -Lample, Guillaume, et al. 2016. "Neural architectures for named entity recognition." arXiv preprint arXiv:1603.01360.

  -Peters, Matthew E., et al. 2018. "Deep contextualized word representations." arXiv preprint arXiv:1802.05365.

  -Hironsan. 2019. Source code from Hironsan/dependabot/pip/werkzeug-0.15.3 [Source code] 
     https://github.com/Hironsan/anago
