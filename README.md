# Program za učenje modelov tipa BERT

V repozitoiju je na voljo program uporabljen pri diplomski nalogi za učenje modela CroSloEngual BERT in SloBERTa.

Že naučen model CroSloEngual BERT smo pridobili iz https://www.clarin.si/repository/xmlui/handle/11356/1330 in ga dali v mapo /cro-slo-eng-bert. 

Že naučen model SloBERTa pa smo pridobili iz 
https://www.clarin.si/repository/xmlui/handle/11356/1397 in ga dali v mapo /sloberta

Podatki za validacijo so v datoteki `datasetValid.txt` in za testiranje v datoteki `datasetTest.txt`.
Podatke za učenje smo razdelili na 2 datoteki `datasetTrainPart1.txt` in `datasetTrainPart2.txt` ker GitHub ne podpira datotek večjih od 100 MB.
Datoteki je potrebno združiti v datoteko `datasetTrain.txt`, saj je uporabljena v kodi.
Podatki so že pripravljeni za uporabo pri učenju.

Pri obeh modelih smo uporabili verzijo za knjižnici Transformers in Pytrorch. 
Učenje modela CroSloEngual BERT smo pognali z ukazom:

`python train.py --cuda=True --pretrained-model=./cro-slo-eng-bert --lr=5e-6 --epoch=6 --save-path=outBert --batch-size=32 --sequence-length=256`

Učenje modela SloBERTa pa smo pognali z ukazom:

`python train.py --cuda=True --pretrained-model=./sloberta --lr=5e-6 --epoch=6 --save-path=outSloberta --batch-size=32 --sequence-length=256`
