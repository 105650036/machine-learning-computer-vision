# machine-learning-computer-vision

目標:透過機器學習去辨識辛普森家庭的20位角色 資料來源:上課老師提供 並要參與kaggle競賽

以下為目標20位角色

![20_characters_illustration](https://user-images.githubusercontent.com/49279418/106090882-30e7fb00-6166-11eb-874b-931e56ea2503.png)

程式方塊圖

![1](https://user-images.githubusercontent.com/49279418/106090955-5248e700-6166-11eb-8820-79c6dc03d999.png)

由於此為分類問題 所以使用one hot 的標籤來當作答案 以下為分類的狀況(考慮輸入大小不同統一改為128X128 且考慮樣本數不一樣多 所以每個角色只取20人)

![2](https://user-images.githubusercontent.com/49279418/106091088-9d62fa00-6166-11eb-9164-dc6605532923.png)

並且配合one hot 把圖片也做轉檔(轉成要的大小)
![3](https://user-images.githubusercontent.com/49279418/106091321-211ce680-6167-11eb-8526-e4d987c16414.png)


以下是模型(使用論文CNN的模型 最後接上全連接層輸出維度為20(用one hot分類有20位角色))
![m1](https://user-images.githubusercontent.com/49279418/106091883-3cd4bc80-6168-11eb-9952-1bfe35028882.png)
![m2](https://user-images.githubusercontent.com/49279418/106091890-3f371680-6168-11eb-9c38-59bceb582789.png)

以下是訓練結果及排名

![r1](https://user-images.githubusercontent.com/49279418/106091897-41997080-6168-11eb-84e5-a815b4bb001f.png)
![r2](https://user-images.githubusercontent.com/49279418/106091898-41997080-6168-11eb-9265-314038536b7a.png)


在輸入其實是可以透過旋轉縮放 或改變顏色來使輸入多樣化 但看的出其實由於角色不易混淆 導致不需做此加工 就可以得到很好的結果了(在測試集有99%正確率 測試集也是接近99%)
