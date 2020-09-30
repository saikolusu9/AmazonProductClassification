#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 12:27:56 2020r

@author: saikolusu
"""

import os
import configparser as cp
from scripts.train.read_data import ReadData
from scripts.train.clean_data import CleanData
from scripts.train.feature_engineering import FeatureExtraction
from scripts.train.train_test_split import Split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

class ExecutePrediction:

    def __init__(self):
        self.path  = "/Users/saikolusu/Documents/Text Mining/ArticlePredicition/AmazonProductClassification/"
        self.cur_path = os.path.dirname(__file__)
        self.config = cp.RawConfigParser()
        #self.properties_file_path = os.path.relpath('../resources/application.properties', self.cur_path)
        self.properties_file_path = (self.path + "scripts/resources/application.properties")
        self.config.read(self.properties_file_path)
        self.data_folder = self.config.get('Properties', 'folder')
        self.data_file = self.config.get('Properties', 'file_name')
        self.columns = ['Category', 'Description']
        self.readdata = ReadData()
        self.cleaneddata = CleanData()
        self.features_extract = FeatureExtraction()
        self.split_df = Split()

    def read_data(self):
        # df = self.readdata.read_data(os.path.relpath('../../' + self.data_folder, self.cur_path), self.data_file, self.columns)
        df = self.readdata.read_data(self.path + self.data_folder, self.data_file, self.columns)

        return df

    def clean_data(self, df, column, tfidf=None, status=True):
        df["cleaned"] = df[column].astype(str).apply(self.cleaneddata._clean)
        if status == False:
            word_vectors_tfidf = tfidf.transform(df["cleaned"].values)
            return tfidf, word_vectors_tfidf
        else:
            tfidf, word_vectors_tfidf = self.features_extract.featrue_extract(df)
            target = self.features_extract.label_encoding(df)
            return tfidf, word_vectors_tfidf, target

    def prediction(self, predict_df):
        train_df = self.read_data()
        word_tfidf, word_vectors_tfidf, target = self.clean_data(train_df,"Description")
        id_to_category = self.features_extract.df_with_targets_actuals(train_df, target)
        train_x, val_x, train_y, val_y = self.split_df.train_test_split(word_vectors_tfidf, target)

        rf = RandomForestClassifier(n_estimators=50, max_features='sqrt', max_depth=100, min_samples_leaf=1,
                                    min_samples_split=2, bootstrap=False, random_state=42)

        rf = self.rf.build_RF()

        # Fit the random search model
        rf.fit(train_x, train_y)

        tfidf, predict_wv = self.clean_data(predict_df,"Text",tfidf=word_tfidf,status=False)
        prediction = rf.predict(predict_wv)[0]

        prediction = id_to_category[prediction]

        return prediction


if __name__ == '__main__':
    status = True
    #message = "In all my years of testing smartphones, I can hardly remember a time I asked myself, Wait, when did I charge this again?That's what Motorola's latest, the One 5G, will make you think. It's a heavy and thick phone, stuffed with a 5,000-mAh battery cell. But I can take some bulk for a phone I don't have to plug in every night.The One 5G is also an example of why you don't need to spend $1,000 to get a good phone. It joins a wave of sub-$500 devices that offer almost everything you need without stuttery performance or terrible cameras (the two most common flaws on cheap phones). There are still some compromises here, but if you want a phone that lasts well more than a day, you'll be hard-pressed to find something better.From the back, the Motorola One 5G resembles the iPhone 11 Pro. Motorola mimics Apple's camera setup, but instead of three cameras, there are four, giving it a more symmetrical design. It looks cluttered, especially with the micro-pattern on the shiny, smudge-attracting plastic back. It's good to see Motorola using plastic—it doesn't feel cheap, and you don't have to worry about it shattering after an accidental drop.The beefy battery makes it chunky, but the narrow form makes it easy to hold, like a TV remote. Even if you have large hands like me, you'll still have trouble reaching the top of the 6.7-inch LCD screen. It's tall and narrow because it has a cinematic aspect ratio: If you lay it horizontally, it's wide enough that you won't see black bars above and below most movies. TV shows, however, aren't shot as wide, so do expect to see black bars on the left and right.Speaking of the screen, it's sharp and gets bright enough to see clearly outdoors, but colors aren't too vibrant, and you don't get the inky blacks of the OLED display in Google's $350 Pixel 4A. It still looks modern with two floating, punchhole selfie cameras at the top left and slim edges all around.This phone's ace in the hole is support for a 90-Hz refresh rate. The display can flip through 90 images per second, so everything looks smoother and more fluid than the 60-Hz screens on many phones.The real spotlight is the 5,000-mAh battery. I almost always recharge test phones at the end of the night so I can continue my doomscrolling the next morning. I didn't bother to do that with the One 5G because it usually had more than 60 percent at the end of the day. At near midnight on the second day, I had around 30 percent in the tank—that's when I plugged in the USB-C charger. These results are with average use, and I easily hit around six and a half hours of screen-on-time. You should get more than a day out of this phone under almost any circumstances.It's fast too. Inside is Qualcomm's Snapdragon 765G, the same chip powering the likes of the LG Velvet and OnePlus Nord. It's not as smooth as the Nord, but it's not as stuttery as the Velvet. If it had a little more than 4 gigabytes of RAM, that might speed it up—most Android phones utilize 6 gigabytes nowadays, including the cheaper Pixel 4A. A few small stutters here and there are not a big deal.Other goodies include a MicroSD card slot if you need to go past the 128 GB of built-in storage, a side-mounted fingerprint sensor, and Motorola has enabled its NFC sensor—something it routinely disables on its affordable US phones to encourage you to buy more expensive models, I imagine. NFC is what allows you to use tap-to-pay services like Google Pay. I've routinely used it during the pandemic to minimize touching other surfaces. I can leave the wallet at home too!The biggest disappointment? The speaker. There's only one, which fires out from the bottom, and it's super easy to block with your hand. It sounds OK, but if you're in a noisy environment, you'll want to plug into the headphone jack (yes, there is one). There's also no IP-rated water resistance (Motorola claims the phone can handle rain and splashes) or wireless charging, but that's common at this price point.Having a lot of cameras on a phone can be a good thing. It means being able to shoot with different perspectives, letting those creative juices flow. But I'd rather have one very good camera than six average ones. The Motorola One 5G packs a main 48-megapixel camera, a 5-megapixel macro, an 8-megapixel ultrawide, a depth camera (for measuring depth in Portrait mode), and two selfie cameras—an 8-MP main and 16-MP ultrawide. They're all ... decent. I would have preferred Motorola cut two of the cameras if it meant a lower price.Photos from the main camera are solid during the day, with muted colors and little contrast, though I have noticed smearing in some of the details. It also can struggle managing high-contrast scenes, sometimes blowing out the sky to expose the foreground. Use Motorola's Night mode in low light and the results are usable if you make sure to stay very still during capture, though it still can't match the quality of the Google Pixel 4A's Night Sight. Portrait mode produces a nice blur effect behind subjects.The ultrawide camera is less impressive. It can take some scenic photos, but the focus is often soft and colors aren't quite as accurate. It also doesn't support Motorola's Night mode, so you won't get good results at night. The macro camera allows you to take super close-ups of anything, and it even has a ring light around it when it detects there isn't enough light. In broad daylight, you can get some fun shots, but the light isn't strong enough for nighttime macro photography.Moto One 5G, ultrawide camera. The ultrawide lets me take in more of the Williamsburg bridge. Zoom in a little and you'll see a lot of grain. It can take nice selfies—even decent low-light ones—because the main front camera supports Night mode. Unfortunately, I don't think the extra front ultrawide camera was necessary. I haven't tested it extensively because who travels in groups these days, but the few selfies I took with my partner outside (in sunny conditions) looked terrible, with fuzzy details and lots of grain. It does not work with Night mode, so don't bother switching to it after hours.There are three big problems with the Motorola One 5G. First, you can't buy it unlocked. It's only available from AT&T (now) and Verizon (in October). Second, there's the 5G in its name. I never encountered the next-gen network in my NYC neighborhood, and you probably won't either. Even if you have a compatible data plan, don't expect to access it soon. It's a nice perk that futureproofs the phone in case your area does get 5G, but not a reason to buy the phone.And third, after Motorola updates this phone to Android 11 (which Google released in early September), you won't get another Android version upgrade. Most phones get two upgrades, so it's disappointing (but unsurprising) to see Motorola fall short here. At the very least, you will get two years of security updates.If you can get past those hurdles, the One 5G is a good phone for the money. If you're a shutterbug, buy the Pixel 4A. But if you prioritize a big screen and battery, you'll be happy here."
    message = "I want a bath soap"
    ex = ExecutePrediction()
    df = pd.DataFrame(np.array([message]), columns=['Text'])
    ret = ex.prediction(df)