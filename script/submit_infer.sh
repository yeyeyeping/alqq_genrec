mkdir -p tmp_infer/

cp utils.py tmp_infer/
cp -r model tmp_infer/
cp -r const tmp_infer/
cp dataset.py tmp_infer/
cp -r mm_emb_loader.py tmp_infer/

zip -r submit_infer.zip tmp_infer

rm -rf tmp_infer