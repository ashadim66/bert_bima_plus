from preprocessing_data import bima_preprocess
from classifier_model import klasifikasi_bima
import streamlit as st


st.title('Klasifikasi Sentimen dari Ulasan Aplikasi Bima+')
st.write('Tentukan sentimen dari ulasan yang diperoleh ')


contoh_teks = st.selectbox(
    'Contoh teks review / ulasan aplikasi Bima+: ',
    ('Harga ramah dikantong ajar', 'Jaring tri bagus internetan bima puas tawar sedia tapi harga paket sedia naik kuota turun gin guna puas bima tri pilih kartu pradana tawar muas', 'Jelek jaring download susah pulsa gua habis beli paket unlimited download susah jelek pokok jelekkkkkk', 'Bagus segi paket spesial kota tingkat'))

if contoh_teks != None:
    processed_example = bima_preprocess(contoh_teks)
    hasil_sentimen, persentase = klasifikasi_bima(processed_example)
    st.write(f"Sentimen dari '{contoh_teks}' adalah: :blue[{hasil_sentimen}] dengan tingkat persentase = :blue[{persentase}]",
             )
else:
    print('Maaf, ada masalah')

st.subheader('Masukkan :blue[ulasan] yang anda peroleh ke bawah:')

ulasan = st.text_area('Teks Ulasan:', )

if st.button('Prediksi Sentimen'):
    processed_sentence = bima_preprocess(ulasan)
    # Jika hasil preprocessing berupa kalimat kosong
    if len(processed_sentence) == 0:
        st.write(f"Tidak bisa melakukan klasifikasi. Kalimat yang anda masukkan terlalu singkat atau hanya berisi angka, emoticon atau kata-kata stopword.")
        st.write(f"Ulasan anda: {ulasan}")
    else:
        hasil_sentimen_bima, persentase_bima = klasifikasi_bima(
            processed_sentence)
        st.write(f"Teks yang dimasukkan:",
                 )
        st.write(f"'{ulasan}'")
        st.write(
            f"Sentimen: :blue[{hasil_sentimen_bima}] dengan tingkat persentase = :blue[{persentase_bima}]")

else:
    st.write('Belum ada ulasan yang masuk')
