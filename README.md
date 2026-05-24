```python
sorted_stats = sorted(zip(radar_labels, radar_values), key=lambda x: x[1])

# nilai dan kategori terendah pertama
kategori_terendah = sorted_stats[0][0] 
nilai_terendah = sorted_stats[0][1]

# nilai dan kategori terendah kedua
kategori_terendah_1 = sorted_stats[1][0] 
nilai_terendah_1 = sorted_stats[1][1]

output = f"Tim Anda memiliki kekurangan di sektor {kategori_terendah} ({nilai_terendah:.2f}) dan {kategori_terendah_1} ({nilai_terendah_1:.2f}). Anda disarankan untuk mengoptimalisasi sektor-sektor tersebut untuk mendapatkan tim yang lebih baik secara statistik."
```

```python
# diunduh di terminal
pip install google-generativeai

# import di file .py
import google.generativeai as genai
```