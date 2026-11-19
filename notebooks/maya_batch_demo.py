# Maya 2 Native BATCH RENDER  -  renders every line of one or more audio dramas on a free Kaggle GPU.
# Upload as a Kaggle notebook (File -> Import).  Session options:  Accelerator = GPU T4 x1  |  Internet = ON  |  FRESH session.
# Output: /kaggle/working/maya_batch.zip  ->  download, then on your laptop:   python -m tools.import_maya_zip maya_batch.zip
# After that the stories play with Maya voices with NO GPU and NO internet (pick "Maya" in the UI).
#
# %% 1. environment check
import json, os, subprocess, sys, time

print(subprocess.run("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv", shell=True, capture_output=True, text=True).stdout)
print("python", sys.version)

# %% 2. jobs (made by tools/export_maya_jobs.py: key = cache file name, voice, language, text)
_RAW = r'''[
{
"key": "me3fbe81b1845966712dd",
"voice": "Anirban",
"lang": "en",
"text": "Scene 1. home.",
"speaker": "Narrator",
"scene": "scene_01"
},
{
"key": "m0c050b4f21d2608149cf",
"voice": "Charvi",
"lang": "hi",
"text": "भैया, नानी का जन्मदिन अगले हफ्ते है ना? मुझे बहुत खुशी हो रही है!",
"speaker": "Ananya",
"scene": "scene_01"
},
{
"key": "mc3164e0faa083902048e",
"voice": "Satvik",
"lang": "hi",
"text": "हाँ, मैं उनसे बहुत प्यार करता हूँ। उनके बिना घर का माहौल ही अलग लगता है।",
"speaker": "Vikram",
"scene": "scene_01"
},
{
"key": "m255f3c9aac61f8b32e86",
"voice": "Charvi",
"lang": "hi",
"text": "इस बार हम नानी को एक खास गिफ्ट देंगे, बहुत अच्छा लगेगा देखकर। अरे, मैं बहुत एक्साइटेड हूँ!",
"speaker": "Ananya",
"scene": "scene_01"
},
{
"key": "m1c99da87edf4992721eb",
"voice": "Satvik",
"lang": "hi",
"text": "ठीक है यार, मैं केक लाऊंगा। तू डेकोरेशन की ज़िम्मेदारी संभाल लेना।",
"speaker": "Vikram",
"scene": "scene_01"
},
{
"key": "m042a3e4c1dbf74dd458e",
"voice": "Charvi",
"lang": "hi",
"text": "पक्का, भैया। मैं सब कुछ चुटकी में सेट कर दूंगी।",
"speaker": "Ananya",
"scene": "scene_01"
},
{
"key": "m8f455e1de3292727077d",
"voice": "Satvik",
"lang": "hi",
"text": "अच्छा, फिर यह प्लान पक्का है। हम दोनों मिल कर यह कर लेंगे।",
"speaker": "Vikram",
"scene": "scene_01"
},
{
"key": "mfb46f6a01699ad4c7699",
"voice": "Anirban",
"lang": "en",
"text": "Scene 2. market.",
"speaker": "Narrator",
"scene": "scene_02"
},
{
"key": "m581ff6a1c188e70a9ddb",
"voice": "Satvik",
"lang": "hi",
"text": "आज शाम बाज़ार में एक बहुत सुंदर केक दिखा। लेकिन उसकी कीमत बहुत ज़्यादा है, मुझे बुरा लग रहा है।",
"speaker": "Vikram",
"scene": "scene_02"
},
{
"key": "m9d2592b500584123e4c9",
"voice": "Siddhant",
"lang": "hi",
"text": "बेटा, इससे सस्ता केक भी मिल जाएगा। वह भी बहुत अच्छा है।",
"speaker": "Dukaandaar",
"scene": "scene_02"
},
{
"key": "m376b5891e90499ba873e",
"voice": "Satvik",
"lang": "hi",
"text": "नहीं, नानी के लिए कुछ अच्छा होना चाहिए। मुझे इस केक की प्रॉब्लम नहीं, बस पैसे की फुल टेंशन है।",
"speaker": "Vikram",
"scene": "scene_02"
},
{
"key": "m8e826155953e07001763",
"voice": "Siddhant",
"lang": "hi",
"text": "सोच लो बेटा, बजट के हिसाब से चुज़ करना अच्छा रहता है।",
"speaker": "Dukaandaar",
"scene": "scene_02"
},
{
"key": "mfecc0e7c612a10aab679",
"voice": "Satvik",
"lang": "hi",
"text": "ठीक है, मैं अपनी सारी बचत लगा दूंगा। नानी का बर्थडे खास होना चाहिए।",
"speaker": "Vikram",
"scene": "scene_02"
},
{
"key": "m862ae08dc112c72b341a",
"voice": "Satvik",
"lang": "hi",
"text": "यह केक मैं ले लेता हूँ, मुझे बहुत खुशी होगी।",
"speaker": "Vikram",
"scene": "scene_02"
},
{
"key": "mdf8f9fc002ddb8879313",
"voice": "Anirban",
"lang": "en",
"text": "Scene 3. Inside the house, near a lit tray.",
"speaker": "Narrator",
"scene": "scene_03"
},
{
"key": "m8c1f52f0ee95d2f104e6",
"voice": "Charvi",
"lang": "hi",
"text": "भैया, बारिश बहुत तेज़ है और बिजली भी चली गई! सब सामान भीग जाएगा, मुझे बहुत डर लग रहा है।",
"speaker": "Ananya",
"scene": "scene_03"
},
{
"key": "m456121a41356b87df2bd",
"voice": "Satvik",
"lang": "hi",
"text": "तू चिंता मत कर, अनन्या। हम दोनों मिलकर सब कुछ ठीक कर लेंगे।",
"speaker": "Vikram",
"scene": "scene_03"
},
{
"key": "m67f330586b5cf3e1ff87",
"voice": "Charvi",
"lang": "hi",
"text": "पर भैया, सजावट का सारा सामान खराब हो गया तो? यह बहुत बुरा होगा।",
"speaker": "Ananya",
"scene": "scene_03"
},
{
"key": "m80e06dfe100a8e5a9faa",
"voice": "Satvik",
"lang": "hi",
"text": "चल, टॉर्च ले आ। सारा सामान ऊपर ट्रे पर रखते हैं, फिर सब सुरक्षित रहेगा।",
"speaker": "Vikram",
"scene": "scene_03"
},
{
"key": "m5d786464ab1e42d87663",
"voice": "Charvi",
"lang": "hi",
"text": "ठीक है भैया। मुझे अब भी डर लग रहा है, पर आप साथ हो तो हिम्मत आ रही है।",
"speaker": "Ananya",
"scene": "scene_03"
},
{
"key": "m827fb71bacb3d882a6fc",
"voice": "Satvik",
"lang": "hi",
"text": "बस, अब घबरा मत। हम साथ हैं, सब एकदम ऑन पॉइंट होगा।",
"speaker": "Vikram",
"scene": "scene_03"
},
{
"key": "md6dcdd63cd42a7f6f183",
"voice": "Anirban",
"lang": "en",
"text": "Scene 4. the house (after returning from the temple).",
"speaker": "Narrator",
"scene": "scene_04"
},
{
"key": "me5a08dc25d559830ed71",
"voice": "Charvi",
"lang": "hi",
"text": "भैया, नानी पूजा के लिए मंदिर चली गयी हैं, तो हम जल्दी से घर सजाते हैं!",
"speaker": "Ananya",
"scene": "scene_04"
},
{
"key": "m7c87b0ec5df0c9c0ef2a",
"voice": "Satvik",
"lang": "hi",
"text": "हाँ, ठीक है। यह फूल और गुब्बारे कहाँ रखने हैं?",
"speaker": "Vikram",
"scene": "scene_04"
},
{
"key": "mdfb53fb57c3169a83894",
"voice": "Jahnavi",
"lang": "hi",
"text": "मैं आ गयी हूँ, मदद करने! गुब्बारे दरवाज़े के पास लगा दो।",
"speaker": "Priya",
"scene": "scene_04"
},
{
"key": "m48b13ddfa1944dae333d",
"voice": "Charvi",
"lang": "hi",
"text": "प्रिया, तुम यह देखो, फूलों से कितना सुंदर लग रहा है! मुझे एकदम लाजवाब लगा।",
"speaker": "Ananya",
"scene": "scene_04"
},
{
"key": "mc61f0cdec80962df35f6",
"voice": "Jahnavi",
"lang": "hi",
"text": "सच में, सब कुछ बहुत सुंदर बन रहा है। देखना, नानी यह सब देखकर बहुत खुश होंगी।",
"speaker": "Priya",
"scene": "scene_04"
},
{
"key": "mc208d1d1376540f435ca",
"voice": "Charvi",
"lang": "hi",
"text": "हाँ, मेरा दिल बहुत ज़ोर से धड़क रहा है, इतनी खुशी हो रही है मुझे!",
"speaker": "Ananya",
"scene": "scene_04"
},
{
"key": "mc945b72d1583775f5c65",
"voice": "Anirban",
"lang": "en",
"text": "Scene 5. Inside Nani's house.",
"speaker": "Narrator",
"scene": "scene_05"
},
{
"key": "m4ed182cfdd8ee8cdf3b0",
"voice": "Charvi",
"lang": "hi",
"text": "नानी, आप आ गयीं! आज का दिन बहुत खास है।",
"speaker": "Ananya",
"scene": "scene_05"
},
{
"key": "m479d7b2f7e50172db228",
"voice": "Malini",
"lang": "hi",
"text": "आज बस इतना हो कि तुम सब मेरे साथ हो, बेटा।",
"speaker": "Nani",
"scene": "scene_05"
},
{
"key": "mcd2648e823c5beb08a3d",
"voice": "Satvik",
"lang": "hi",
"text": "नानी, आप दरवाज़ा खोलते ही रुक गयीं, क्या हुआ?",
"speaker": "Vikram",
"scene": "scene_05"
},
{
"key": "m31b3fcf253161f280999",
"voice": "Malini",
"lang": "hi",
"text": "इतना सुंदर घर देखकर मैं रुक गयी, बेटा। और सामने तुम सब खड़े हो।",
"speaker": "Nani",
"scene": "scene_05"
},
{
"key": "m3fef82a170928881f442",
"voice": "Malini",
"lang": "hi",
"text": "मेरी आँखों में आँसू आ गए हैं, और मैं तुम तीनों को गले लगाती हूँ। यह मेरी ज़िंदगी का सबसे अच्छा तोहफ़ा है। पूरा घर हँसी और ख़ुशी से भर गया है।",
"speaker": "Nani",
"scene": "scene_05"
},
{
"key": "m2954359fba4ec76d218a",
"voice": "Charvi",
"lang": "hi",
"text": "इस बार हम नानी को एक खास गिफ्ट देंगे, बहुत अच्छा लगेगा देखकर। मैं बहुत एक्साइटेड हूँ!",
"speaker": "Ananya",
"scene": "scene_01"
},
{
"key": "mf95dd45fde9fd640901a",
"voice": "Satvik",
"lang": "hi",
"text": "ठीक है, मैं केक लाऊँगा। तू डेकोरेशन की ज़िम्मेदारी संभाल लेना।",
"speaker": "Vikram",
"scene": "scene_01"
},
{
"key": "me0b2d98a7e557bbf9913",
"voice": "Charvi",
"lang": "hi",
"text": "पक्का, भैया। मैं सब कुछ आसानी से सेट कर दूँगी।",
"speaker": "Ananya",
"scene": "scene_01"
},
{
"key": "m9f541ff3301ee9d60e61",
"voice": "Satvik",
"lang": "hi",
"text": "नहीं, नानी के लिए कुछ अच्छा होना चाहिए। मुझे इस केक की प्रॉब्लम नहीं, बस पैसों की चिंता है।",
"speaker": "Vikram",
"scene": "scene_02"
},
{
"key": "m47d8501369aeea2b85cc",
"voice": "Satvik",
"lang": "hi",
"text": "बस, अब घबरा मत। हम साथ हैं, सब अच्छा होगा।",
"speaker": "Vikram",
"scene": "scene_03"
},
{
"key": "m44194822dce768e53fe3",
"voice": "Charvi",
"lang": "hi",
"text": "प्रिया, तुम यह देखो, फूलों से कितना सुंदर लग रहा है! मुझे बहुत अच्छा लगा।",
"speaker": "Ananya",
"scene": "scene_04"
},
{
"key": "m8340167edd196e90326c",
"voice": "Anirban",
"lang": "en",
"text": "Scene 1. The school cricket field where the team gathers.",
"speaker": "Narrator",
"scene": "scene_01"
},
{
"key": "mae0edc8b273d24227f00",
"voice": "Nakul",
"lang": "hi",
"text": "सर, आज हमारा टूर्नामेंट अनाउंस हो गया है, मुझे बहुत एक्साइटमेंट हो रही है!",
"speaker": "Kabir",
"scene": "scene_01"
},
{
"key": "m8e3210adb2196f31ae48",
"voice": "Siddhant",
"lang": "hi",
"text": "बिलकुल कबीर, अगले महीने इंटर-स्कूल टूर्नामेंट है और हमें हर हाल में फाइनल्स तक पहुँचना है।",
"speaker": "Coach Varma",
"scene": "scene_01"
},
{
"key": "m539c785718c8b8023b61",
"voice": "Nakul",
"lang": "hi",
"text": "सर, हम अपनी पूरी जान लगा देंगे, मुझे पूरा भरोसा है कि हम कर सकते हैं।",
"speaker": "Kabir",
"scene": "scene_01"
},
{
"key": "m618cef152420fe074626",
"voice": "Satvik",
"lang": "hi",
"text": "कबीर यार, इस बार कप हमारा ही है, मैं बहुत खुश हूँ कि हम साथ में खेल रहे हैं।",
"speaker": "Rohan",
"scene": "scene_01"
},
{
"key": "m2018bffa39f745f57930",
"voice": "Charvi",
"lang": "hi",
"text": "मैं स्कोरकार्ड बहुत ध्यान से रखूँगी, पर मुझे थोड़ी चिंता है कि प्रैक्टिस टाइम कम है।",
"speaker": "Mira",
"scene": "scene_01"
},
{
"key": "m1e8fc3bd6fac6b7a3960",
"voice": "Ahana",
"lang": "hi",
"text": "भैया, मैं हर मैच में तालियाँ बजाने आऊँगी, आप सब बहुत अच्छा खेलना!",
"speaker": "Sneha",
"scene": "scene_01"
},
{
"key": "m457fbb9cf70ee6ca1c82",
"voice": "Anirban",
"lang": "en",
"text": "Scene 2. Cricket practice field.",
"speaker": "Narrator",
"scene": "scene_02"
},
{
"key": "m0ed973c73c4131bdf688",
"voice": "Nakul",
"lang": "hi",
"text": "रोहन, मैं रोज़ सुबह पाँच बजे उठकर मैदान पहुँच जाता हूँ, और शाम तक पसीना बहता रहता है। बहुत थकान होती है, लेकिन मुझे एकदम लाजवाब लग रहा है।",
"speaker": "Kabir",
"scene": "scene_02"
},
{
"key": "m3fc328fb21e47dfc03d7",
"voice": "Satvik",
"lang": "hi",
"text": "हाँ यार, तब ही तो तुम सबसे तेज़ दौड़ रहे हो। मुझे भी कल से बहुत मेहनत करनी है।",
"speaker": "Rohan",
"scene": "scene_02"
},
{
"key": "m0f7262599d1ea67a0978",
"voice": "Satvik",
"lang": "hi",
"text": "आह! मेरा पैर... बहुत दर्द हो रहा है, मैं सही से खड़ा नहीं हो पा रहा!",
"speaker": "Rohan",
"scene": "scene_02"
},
{
"key": "m9829b65052e56fa9dd63",
"voice": "Charvi",
"lang": "hi",
"text": "रोहन! तुम ठीक हो ना? मुझे बहुत डर लग रहा है, कुछ ज़्यादा तो नहीं हुआ?",
"speaker": "Mira",
"scene": "scene_02"
},
{
"key": "m45a1c63681b05d8531d9",
"voice": "Satvik",
"lang": "hi",
"text": "चिंता मत करो मीरा, बस थोड़ी सी मोच है। मैं ठीक हो जाऊँगा।",
"speaker": "Rohan",
"scene": "scene_02"
},
{
"key": "m4583dac605c17d78aab5",
"voice": "Nakul",
"lang": "hi",
"text": "यह एकदम वाहियात हुआ। टूर्नामेंट से पहले ही हमारा सबसे बड़ा खिलाड़ी घायल हो गया, मुझे बहुत खराब लग रहा है।",
"speaker": "Kabir",
"scene": "scene_02"
},
{
"key": "mf5c1e9b43ab8c8dd30af",
"voice": "Anirban",
"lang": "en",
"text": "Scene 3. Dadaji's house, living room.",
"speaker": "Narrator",
"scene": "scene_03"
},
{
"key": "m7da03272315200b01253",
"voice": "Nakul",
"lang": "hi",
"text": "दादाजी, आप यहाँ बैठे हैं? सब ठीक है ना?",
"speaker": "Kabir",
"scene": "scene_03"
},
{
"key": "me1e3ace19db847b7b454",
"voice": "Vasant",
"lang": "hi",
"text": "हाँ बेटा, तेरा ही इंतज़ार कर रहा था। इतना उदास क्यों लग रहा है?",
"speaker": "Dadaji",
"scene": "scene_03"
},
{
"key": "m8ff6389a5d3805ffec1e",
"voice": "Nakul",
"lang": "hi",
"text": "दादाजी, रोहन को चोट लग गई है। मुझे फुल टेंशन हो रही है, लगता है हम टूर्नामेंट हार जाएंगे।",
"speaker": "Kabir",
"scene": "scene_03"
},
{
"key": "m1c7a0d3a716d1046efa8",
"voice": "Vasant",
"lang": "hi",
"text": "बेटा, हार-जीत तो खेल का हिस्सा है। हिम्मत कभी मत हारना।",
"speaker": "Dadaji",
"scene": "scene_03"
},
{
"key": "m6edac890efee5d5eea7b",
"voice": "Jahnavi",
"lang": "hi",
"text": "बेटा, तू बस अपना सर्वश्रेष्ठ दे। बाकी भगवान पर छोड़ दे।",
"speaker": "Mummy",
"scene": "scene_03"
},
{
"key": "m525d3c55a9113929b2a7",
"voice": "Nakul",
"lang": "hi",
"text": "ठीक है मम्मी, मैं कोशिश करूंगा।",
"speaker": "Kabir",
"scene": "scene_03"
},
{
"key": "m62483c3b79ff5558b751",
"voice": "Anirban",
"lang": "en",
"text": "Scene 4. school.",
"speaker": "Narrator",
"scene": "scene_04"
},
{
"key": "mf695ace9d65c6383bf08",
"voice": "Ahana",
"lang": "hi",
"text": "भैया, देखो, मैंने एक नया नाम सुना है। नौवीं क्लास का आर्यन बहुत अच्छी गेंदबाजी करता है। उसे टीम में ले लो ना!",
"speaker": "Sneha",
"scene": "scene_04"
},
{
"key": "m5d4510070ad39f8d9e5e",
"voice": "Nakul",
"lang": "hi",
"text": "स्नेहा, यह तो एकदम कमाल लग रहा है। लेकिन आर्यन सर की टीम में नहीं है, वो दूसरे स्कूल का खिलाड़ी है।",
"speaker": "Kabir",
"scene": "scene_04"
},
{
"key": "m6389c0dc77d95fe7764b",
"voice": "Ahana",
"lang": "hi",
"text": "हां, लेकिन खेल में फर्क क्या पड़ता है? वो सच में काफी अच्छा है।",
"speaker": "Sneha",
"scene": "scene_04"
},
{
"key": "me595ec8e3104d6b4fd06",
"voice": "Nakul",
"lang": "hi",
"text": "मुझे भी वो पसंद आया, पर रोहन का पुराना दुश्मन भी है। यह एकदम घटिया हो सकता है।",
"speaker": "Kabir",
"scene": "scene_04"
},
{
"key": "m552330f382050035980d",
"voice": "Charvi",
"lang": "hi",
"text": "कबीर, यह तो बड़ी प्रॉब्लम वाली बात है। रोहन को यह बिल्कुल पसंद नहीं आएगा, मुझे बहुत चिंता हो रही है।",
"speaker": "Mira",
"scene": "scene_04"
},
{
"key": "m8ec32de5eed903e0ee7b",
"voice": "Nakul",
"lang": "hi",
"text": "मीरा, मैं समझता हूं। अभी सोचना पड़ेगा।",
"speaker": "Kabir",
"scene": "scene_04"
},
{
"key": "m50945c25a8d88e87aa6c",
"voice": "Anirban",
"lang": "en",
"text": "Scene 5. maidan (field).",
"speaker": "Narrator",
"scene": "scene_05"
},
{
"key": "mee9126c3b429ab55ffed",
"voice": "Nakul",
"lang": "hi",
"text": "रोहन, मुझे तुमसे एक बात करनी है। थोड़ा सा प्रॉब्लम लग रहा है, पर मुझे एकदम कचरा लग रहा है कि हम ऐसे लड़ते रहे।",
"speaker": "Kabir",
"scene": "scene_05"
},
{
"key": "m15c213a6d4b7a1a60b0a",
"voice": "Satvik",
"lang": "hi",
"text": "तुम मुझसे क्या कहना चाहते हो? मुझे आर्यन से जलन होती है, और ये बात मुझे बहुत खराब लगती है।",
"speaker": "Rohan",
"scene": "scene_05"
},
{
"key": "m3a8f725599629d83d3b5",
"voice": "Satvik",
"lang": "hi",
"text": "पर मैं जानता हूं कि टीम के लिए वही सही रहेगा। मैं टीम के साथ हूं, कबीर।",
"speaker": "Rohan",
"scene": "scene_05"
},
{
"key": "m8e2739fd1b6808ba4d40",
"voice": "Nakul",
"lang": "hi",
"text": "तुम सच में बहुत अच्छे दिल के हो, रोहन। मुझे तुम्हारी ये बात सुनकर एकदम फट्टू लगा।",
"speaker": "Kabir",
"scene": "scene_05"
},
{
"key": "mca20b38c5c4b593a3967",
"voice": "Shaunak",
"lang": "hi",
"text": "मैं मुख्य पुरानी बातों के लिए माफ़ी माँगता हूँ। मुझे बहुत शर्म आ रही है। अब हम सब एक ही टीम हैं।",
"speaker": "Aryan",
"scene": "scene_05"
},
{
"key": "m4fa07fba4f50f628d76c",
"voice": "Nakul",
"lang": "hi",
"text": "हाँ, अब हम सब एक ही टीम हैं। यह सुन कर मुझे बहुत खुशी हुई।",
"speaker": "Kabir",
"scene": "scene_05"
},
{
"key": "m793c84b942e5731224ab",
"voice": "Anirban",
"lang": "en",
"text": "Scene 6. Stadium.",
"speaker": "Narrator",
"scene": "scene_06"
},
{
"key": "m1676fd72d2f8dd060c80",
"voice": "Charvi",
"lang": "hi",
"text": "कबीर, स्टेडियम में इतना शोर है, और शुरुआत एकदम फालतू हुई। तीन खिलाड़ी जल्दी आउट हो गए। हम बहुत पीछे चल रहे हैं!",
"speaker": "Mira",
"scene": "scene_06"
},
{
"key": "m75ed6cc2a7f66e84b801",
"voice": "Nakul",
"lang": "hi",
"text": "मीरा, घबराओ मत। मैं गहरी साँस लेता हूँ। अभी आर्यन की गेंदबाज़ी देखते हैं।",
"speaker": "Kabir",
"scene": "scene_06"
},
{
"key": "m9cc11b7335311f685a1c",
"voice": "Shaunak",
"lang": "hi",
"text": "मैंने एक जादुई गेंदबाज़ी की, और चार विकेट लेकर टीम को मैच में वापस ला दिया।",
"speaker": "Aryan",
"scene": "scene_06"
},
{
"key": "m6e75587fbecd36067106",
"voice": "Charvi",
"lang": "hi",
"text": "कबीर, मुझे एकदम ऑन पॉइंट लग रहा है। अब टीम में फिर से उम्मीद आ गई है।",
"speaker": "Mira",
"scene": "scene_06"
},
{
"key": "m6f09a58410e887a6bf25",
"voice": "Nakul",
"lang": "hi",
"text": "आखिरी ओवर में मैं छक्का मारूँगा। स्टेडियम तालियों से गूँज उठेगा।",
"speaker": "Kabir",
"scene": "scene_06"
},
{
"key": "ma0467aa0df10cb823142",
"voice": "Ahana",
"lang": "hi",
"text": "भैया, तुमने तो बहुत अच्छा खेला! मुझे इतनी खुशी हो रही है, तुम तो एक हीरो हो!",
"speaker": "Sneha",
"scene": "scene_06"
},
{
"key": "m2f2f78a916a8b488eb7a",
"voice": "Anirban",
"lang": "en",
"text": "Scene 7. Hospital.",
"speaker": "Narrator",
"scene": "scene_07"
},
{
"key": "me876383442a47137396c",
"voice": "Abhay",
"lang": "hi",
"text": "कबीर, बेटा, दादाजी की तबीयत अचानक बिगड़ गई है। उन्हें तुरंत अस्पताल ले जाना पड़ेगा।",
"speaker": "Papa",
"scene": "scene_07"
},
{
"key": "ma1a95317fbe873331355",
"voice": "Nakul",
"lang": "hi",
"text": "क्या? पापा, आप ठीक हो? मुझे बहुत घबराहट हो रही है।",
"speaker": "Kabir",
"scene": "scene_07"
},
{
"key": "m8f28ba65e236185a1d4c",
"voice": "Ahana",
"lang": "hi",
"text": "भैया, मुझे बहुत डर लग रहा है। दादाजी कैसे हैं?",
"speaker": "Sneha",
"scene": "scene_07"
},
{
"key": "m7f279d705a2c8dc35d10",
"voice": "Jahnavi",
"lang": "hi",
"text": "कबीर, देखो ना, दादाजी की हालत बहुत खराब हो गई थी। मैं बहुत रो रही थी।",
"speaker": "Mummy",
"scene": "scene_07"
},
{
"key": "m285b408f889ad205929c",
"voice": "Akshat",
"lang": "hi",
"text": "घबराने की ज़रूरत नहीं है। दादाजी की हालत अब स्थिर है, बस उन्हें आराम की ज़रूरत है।",
"speaker": "Doctor",
"scene": "scene_07"
},
{
"key": "m796ce73e3b10c8165f58",
"voice": "Nakul",
"lang": "hi",
"text": "दादाजी, मैं आपके पास हूँ। कल का मैच मैं नहीं खेलूँगा, मुझे आपके पास रहना है।",
"speaker": "Kabir",
"scene": "scene_07"
},
{
"key": "m4d1e0ff428a21d1c7ff1",
"voice": "Anirban",
"lang": "en",
"text": "Scene 8. Inside the family home, likely the living area where Kabir is being cared for.",
"speaker": "Narrator",
"scene": "scene_08"
},
{
"key": "m6e5011c7a299f604eccc",
"voice": "Vasant",
"lang": "hi",
"text": "कबीर, बेटा, मेरे पास आओ। मुझे सब याद आ रहा है, तुम मेरा हाथ पकड़े रहो।",
"speaker": "Dadaji",
"scene": "scene_08"
},
{
"key": "m63754cdb2e34ed10b9ff",
"voice": "Nakul",
"lang": "hi",
"text": "दादाजी, आप ठीक हो गए! आपने मुझे इतना डरा दिया था।",
"speaker": "Kabir",
"scene": "scene_08"
},
{
"key": "maeb6a9144b358aa088d0",
"voice": "Vasant",
"lang": "hi",
"text": "पगले, मेरी वजह से अपना सपना मत छोड़ना। मैं तेरा मैच टीवी पर देखूँगा, और तू वहाँ जीतकर आना।",
"speaker": "Dadaji",
"scene": "scene_08"
},
{
"key": "m33651fbdaeea06b078ef",
"voice": "Nakul",
"lang": "hi",
"text": "दादाजी, आप जल्दी ठीक हो जाइए। मैं आपके लिए कप जीतकर लाऊँगा।",
"speaker": "Kabir",
"scene": "scene_08"
},
{
"key": "md3299c4f4629100159a7",
"voice": "Jahnavi",
"lang": "hi",
"text": "बेटा कबीर, रो मत। सब ठीक हो जाएगा। तुम हिम्मत रखो।",
"speaker": "Mummy",
"scene": "scene_08"
},
{
"key": "mde46436e55cfd4298eba",
"voice": "Ahana",
"lang": "hi",
"text": "भैया, हम सब तुम्हारे साथ हैं।",
"speaker": "Sneha",
"scene": "scene_08"
},
{
"key": "m706dbf0e8a6abbec4da0",
"voice": "Anirban",
"lang": "en",
"text": "Scene 9. sports field.",
"speaker": "Narrator",
"scene": "scene_09"
},
{
"key": "m84fdb7513176d6c3357c",
"voice": "Nakul",
"lang": "hi",
"text": "रोहन यार, तुम चोट के बावजूद फाइनल ओवर फेंकने आए और सीधा यॉर्कर डाल दिया! सच में बहुत अच्छा लगा।",
"speaker": "Kabir",
"scene": "scene_09"
},
{
"key": "m327c142bf5a25487c4e5",
"voice": "Satvik",
"lang": "hi",
"text": "दर्द तो था, लेकिन टीम का भरोसा मुझे रोक नहीं पाया। तुम सबने मिल कर जीतने वाला खेल दिखाया।",
"speaker": "Rohan",
"scene": "scene_09"
},
{
"key": "m43e6dd5f54a11ca3034b",
"voice": "Charvi",
"lang": "hi",
"text": "अरे, मैं रो दिया, कबीर! तुम सब ने मिलकर आज इतिहास रच दिया, यह बहुत बड़ी बात है।",
"speaker": "Mira",
"scene": "scene_09"
},
{
"key": "mee346cbf9742f426c96e",
"voice": "Siddhant",
"lang": "hi",
"text": "बच्चों, मुझे तुम पर बहुत गर्व है। आज तुमने साबित किया कि मेहनत और एकता क्या कर सकती है।",
"speaker": "Coach Varma",
"scene": "scene_09"
},
{
"key": "md8b5f793c7b642f762cf",
"voice": "Nakul",
"lang": "hi",
"text": "सर, शुक्रिया। मैं अभी अस्पताल जाकर दादाजी को यह खबर सुनाता हूँ।",
"speaker": "Kabir",
"scene": "scene_09"
},
{
"key": "m2a106ecd234a08c98c04",
"voice": "Vasant",
"lang": "hi",
"text": "वाह, मेरे बेटे! आज तुम्हारी जीत ने मेरा दिन बना दिया।",
"speaker": "Dadaji",
"scene": "scene_09"
},
{
"key": "m70c1957b1d1d0930f2cf",
"voice": "Anirban",
"lang": "en",
"text": "Scene 10. the school field (maidan).",
"speaker": "Narrator",
"scene": "scene_10"
},
{
"key": "m35accfa11a19ace51ddc",
"voice": "Nakul",
"lang": "hi",
"text": "रोहन, एक गोल और चाहिए। सब कुछ अब इसी गेंद पर है।",
"speaker": "Kabir",
"scene": "scene_10"
},
{
"key": "m2d94bde69784b7a83708",
"voice": "Satvik",
"lang": "hi",
"text": "तू टेंशन मत ले, कबीर। मुझे पता है तू कर सकता है।",
"speaker": "Rohan",
"scene": "scene_10"
},
{
"key": "m341fda703cd31937deda",
"voice": "Nakul",
"lang": "hi",
"text": "मैंने आँखें बंद कीं और दादाजी को याद किया। मुझे एकदम झक्कास लगा।",
"speaker": "Kabir",
"scene": "scene_10"
},
{
"key": "m186c7e61a93570d8beb3",
"voice": "Shaunak",
"lang": "hi",
"text": "कबीर! देख, गेंद लाइन पार कर गई! हम जीत गए!",
"speaker": "Aryan",
"scene": "scene_10"
},
{
"key": "m60d2088dcd19f81deb94",
"voice": "Satvik",
"lang": "hi",
"text": "कबीर, तू रो रहा है? आज तू सच में हीरो है!",
"speaker": "Rohan",
"scene": "scene_10"
},
{
"key": "m8bea05dad79c21f4e033",
"voice": "Nakul",
"lang": "hi",
"text": "आर्यन, रोहन, आओ न। हम तीनों साथ हैं, यह जीत हम सब की है।",
"speaker": "Kabir",
"scene": "scene_10"
},
{
"key": "me87dda74284066942c53",
"voice": "Anirban",
"lang": "en",
"text": "Scene 11. Hospital.",
"speaker": "Narrator",
"scene": "scene_11"
},
{
"key": "m5ed6123c9fac351431f0",
"voice": "Nakul",
"lang": "hi",
"text": "दादाजी, मैं कप लेकर सीधा अस्पताल आया हूँ। आपको सबसे पहले दिखाना था।",
"speaker": "Kabir",
"scene": "scene_11"
},
{
"key": "m221fe2d645710c67ac98",
"voice": "Vasant",
"lang": "hi",
"text": "बेटा, तूने आज मेरा सिर गर्व से ऊँचा कर दिया। आँखों में आँसू हैं, पर यह खुशी के आँसू हैं।",
"speaker": "Dadaji",
"scene": "scene_11"
},
{
"key": "mb09e1c82a342e251964e",
"voice": "Nakul",
"lang": "hi",
"text": "दादाजी, यह जीत आपकी है। आपकी दुआओं की वजह से ही हम यहां तक पहुंचे। आपका आशीर्वाद चाहिए।",
"speaker": "Kabir",
"scene": "scene_11"
},
{
"key": "m3206b68f658db2efc328",
"voice": "Ahana",
"lang": "hi",
"text": "दादाजी, बहुत अच्छा लगा यह देखकर! हम सब बहुत खुश हैं।",
"speaker": "Sneha",
"scene": "scene_11"
},
{
"key": "m5bbebc6d36e28fbd9151",
"voice": "Satvik",
"lang": "hi",
"text": "सच में, घर का हर इंसान आज बहुत खुश है। सब एक साथ हैं।",
"speaker": "Rohan",
"scene": "scene_11"
},
{
"key": "m65b097d2e5a4e5560541",
"voice": "Jahnavi",
"lang": "hi",
"text": "आज हमारा परिवार सच में पूरा हो गया है।",
"speaker": "Mummy",
"scene": "scene_11"
},
{
"key": "mdd1dc6191917ad1fe85a",
"voice": "Satvik",
"lang": "hi",
"text": "कबीर, इस बार कप हमारा ही है, मैं बहुत खुश हूँ कि हम साथ में खेल रहे हैं।",
"speaker": "Rohan",
"scene": "scene_01"
},
{
"key": "mf319eadeb81647dcff12",
"voice": "Nakul",
"lang": "hi",
"text": "रोहन, मैं रोज़ सुबह पाँच बजे उठकर मैदान पहुँच जाता हूँ, और शाम तक पसीना बहाता रहता हूँ। बहुत थकान होती है, लेकिन मुझे अच्छा लग रहा है।",
"speaker": "Kabir",
"scene": "scene_02"
},
{
"key": "m619474347acecbdd0ec3",
"voice": "Satvik",
"lang": "hi",
"text": "हाँ, तब ही तो तुम सबसे तेज़ दौड़ रहे हो। मुझे भी कल से बहुत मेहनत करनी है।",
"speaker": "Rohan",
"scene": "scene_02"
},
{
"key": "m1a150dc090dba5f4bc89",
"voice": "Nakul",
"lang": "hi",
"text": "यह बहुत बुरा हुआ। टूर्नामेंट से पहले ही हमारा सबसे बड़ा खिलाड़ी घायल हो गया, मुझे बहुत खराब लग रहा है।",
"speaker": "Kabir",
"scene": "scene_02"
},
{
"key": "m49504c0021237c7195a6",
"voice": "Nakul",
"lang": "hi",
"text": "दादाजी, रोहन को चोट लग गई है। मुझे बहुत चिंता हो रही है, लगता है हम टूर्नामेंट हार जाएँगे।",
"speaker": "Kabir",
"scene": "scene_03"
},
{
"key": "me6f126ea1a4d0150938c",
"voice": "Nakul",
"lang": "hi",
"text": "स्नेहा, यह तो बहुत अच्छा लग रहा है। लेकिन आर्यन सर की टीम में नहीं है, वो दूसरे स्कूल का खिलाड़ी है।",
"speaker": "Kabir",
"scene": "scene_04"
},
{
"key": "mc31e31241c2e8b6c576c",
"voice": "Nakul",
"lang": "hi",
"text": "मुझे भी वो पसंद आया, पर रोहन का पुराना दुश्मन भी है। यह बहुत खराब हो सकता है।",
"speaker": "Kabir",
"scene": "scene_04"
},
{
"key": "m6deceb11919380e83bfe",
"voice": "Nakul",
"lang": "hi",
"text": "रोहन, मुझे तुमसे एक बात करनी है। थोड़ा सा प्रॉब्लम लग रहा है, पर मुझे बहुत बुरा लग रहा है कि हम ऐसे लड़ते रहे।",
"speaker": "Kabir",
"scene": "scene_05"
},
{
"key": "mf8e8346491d8f3d84858",
"voice": "Nakul",
"lang": "hi",
"text": "तुम सच में बहुत अच्छे दिल के हो, रोहन। मुझे तुम्हारी ये बात सुनकर बहुत अच्छा लगा।",
"speaker": "Kabir",
"scene": "scene_05"
},
{
"key": "m71a40050b1adeb5a72c2",
"voice": "Charvi",
"lang": "hi",
"text": "कबीर, स्टेडियम में इतना शोर है, और शुरुआत बहुत खराब हुई। तीन खिलाड़ी जल्दी आउट हो गए। हम बहुत पीछे चल रहे हैं!",
"speaker": "Mira",
"scene": "scene_06"
},
{
"key": "m5630b21c00425691a0c5",
"voice": "Charvi",
"lang": "hi",
"text": "कबीर, मुझे बहुत अच्छा लग रहा है। अब टीम में फिर से उम्मीद आ गई है।",
"speaker": "Mira",
"scene": "scene_06"
},
{
"key": "mb8efc3aae87116e25d65",
"voice": "Nakul",
"lang": "hi",
"text": "रोहन, तुम चोट के बावजूद फाइनल ओवर फेंकने आए और सीधा यॉर्कर डाल दिया! सच में बहुत अच्छा लगा।",
"speaker": "Kabir",
"scene": "scene_09"
},
{
"key": "m33eea4e3a0d01f83e376",
"voice": "Charvi",
"lang": "hi",
"text": "मैं रो दिया, कबीर! तुम सब ने मिलकर आज इतिहास रच दिया, यह बहुत बड़ी बात है।",
"speaker": "Mira",
"scene": "scene_09"
},
{
"key": "m98ba22e892261c906362",
"voice": "Nakul",
"lang": "hi",
"text": "मैंने आँखें बंद कीं और दादाजी को याद किया। मुझे बहुत अच्छा लगा।",
"speaker": "Kabir",
"scene": "scene_10"
}
]'''
assert _RAW.lstrip().startswith("["), "this is the TEMPLATE - upload the generated  notebooks/maya_batch_<name>.py  instead"
JOBS = json.loads(_RAW)
print(f"{len(JOBS)} clips to render, voices: {sorted({j['voice'] for j in JOBS})}")

# %% 3. install  (the model card pins torch 2.8.0 / cu128, transformers 5.6.2, speechbrain 1.1.2)
cmd = ('pip install -q "maya3[server] @ git+https://github.com/MayaResearch/maya2-native" '
       "--extra-index-url https://download.pytorch.org/whl/cu128")
r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print(r.stdout[-1500:], r.stderr[-2500:])
assert r.returncode == 0, "install failed - see the log above and send it back"
# Kaggle's torchvision / torchaudio were built for ITS torch; after the swap they crash transformers on import. Maya needs neither.
r = subprocess.run("pip uninstall -y -q torchvision torchaudio", shell=True, capture_output=True, text=True)
if "torch" in sys.modules:
    raise SystemExit("torch was already imported in this session: Run -> Restart session, then run all cells again.")

# %% 4. load the model
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "no GPU: turn on Accelerator -> GPU T4 x1 in the session options"
from maya3 import Maya3TTS

WORKERS = 4                                             # concurrent requests are batched on the GPU (~0.3 GB per extra seat)
t0 = time.time()
tts = Maya3TTS.from_pretrained("maya-research/maya2-native", max_seats=WORKERS)
print(f"loaded in {time.time() - t0:.0f}s ; GPU memory {torch.cuda.memory_allocated() / 1e9:.1f} GB")

# %% 5. render everything (a failed clip is logged and skipped, never stops the batch)
import numpy as np, soundfile as sf
from concurrent.futures import ThreadPoolExecutor

OUT = "/kaggle/working/maya_batch"
os.makedirs(OUT, exist_ok=True)
SR = 24000
log = {}


def gen(job):
    last = None
    for kw in ({"normalize": True}, {}):
        try:
            return np.asarray(tts.generate(job["text"], lang=job["lang"], voice=job["voice"], **kw), dtype=np.float32)
        except TypeError as e:                          # this release may not take `normalize` on generate()
            last = e
    raise last


def one(job):
    path = f"{OUT}/{job['key']}.wav"
    if os.path.exists(path):
        return
    s = time.time()
    try:
        wav = gen(job)
        if len(wav) < 0.2 * SR:
            raise RuntimeError("audio too short")
        sf.write(path + ".part.wav", wav, SR)
        os.replace(path + ".part.wav", path)
        log[job["key"]] = {"ok": True, "secs_audio": round(len(wav) / SR, 2), "secs_gen": round(time.time() - s, 2)}
    except Exception as e:
        log[job["key"]] = {"ok": False, "error": repr(e)[:200]}
        print("FAILED", job["key"], job.get("speaker"), repr(e)[:160])


t0 = time.time()
try:
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i, _ in enumerate(ex.map(one, JOBS), 1):
            if i % 10 == 0 or i == len(JOBS):
                print(f"  {i}/{len(JOBS)} done, {time.time() - t0:.0f}s")
except Exception as e:                                  # parallel mode is the less-tested path: finish whatever is left one at a time
    print("parallel rendering failed (", repr(e)[:120], ") - continuing sequentially")
    for j in JOBS:
        one(j)
for j in JOBS:                                          # second pass: anything missing is retried once, one at a time
    if not os.path.exists(f"{OUT}/{j['key']}.wav"):
        one(j)
ok = [k for k, v in log.items() if v.get("ok")]
missing = [j["key"] for j in JOBS if not os.path.exists(f"{OUT}/{j['key']}.wav")]
tot_a = sum(log[k]["secs_audio"] for k in ok)
print(f"\nrendered {len(ok)} new clips, {tot_a:.0f}s of audio in {time.time() - t0:.0f}s; missing: {len(missing)}")
json.dump({"log": log, "missing": missing, "n_jobs": len(JOBS)}, open(f"{OUT}/batch_log.json", "w"), indent=1)

# %% 6. zip for download
tts.close()
import shutil
shutil.make_archive("/kaggle/working/maya_batch", "zip", OUT)
print("ready: /kaggle/working/maya_batch.zip  ->  right panel 'Output' -> download   (missing clips:", len(missing), ")")
