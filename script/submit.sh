rm -rf submit
mkdir -p submit

cp -r const/ submit/
cp -r model/ submit/
cp dataset.py submit/
cp loss.py submit/
cp main.py submit/
cp mm_emb_loader.py submit/
cp sampler.py submit/
cp utils.py submit/

zip -r submit.zip submit

rm -rf submit