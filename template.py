from flask import Flask, render_template, request
from bs4 import BeautifulSoup
import requests
import numpy as np
import pandas as pd
import csv

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = request.form["url"]
        result = webscraper(url)
        return render_template("index.html", result=result)
    return render_template("form.html")


def webscraper(url):
    # ใส่โค้ด webscraper ของคุณที่นี่
    global nameteam
    global L
    global b
    url = L.get()

    res = requests.get(url)
    res.encoding = "utf-8"

    soup = BeautifulSoup(res.text, "lxml")

    StoreNumHome = {}
    StoreNumAway = {}
    StoreNumDraw = {}
    StoreNumHomeAfter = {}
    StoreNumAwayAfter = {}
    StoreNumDrawAfter = {}
    StoreOdd1 = {}
    StoreHiLo1 = {}

    StoreOdd2 = {}
    StoreHiLo2 = {}
    for val in range(5):
        StoreNumHome["Case{0}".format(val + 1)] = 0
        StoreNumAway["Case{0}".format(val + 1)] = 0
        StoreNumDraw["Case{0}".format(val + 1)] = 0
        StoreNumHomeAfter["Case{0}".format(val + 1)] = 0
        StoreNumAwayAfter["Case{0}".format(val + 1)] = 0
        StoreNumDrawAfter["Case{0}".format(val + 1)] = 0
        StoreOdd1["Case{0}".format(val + 1)] = 0
        StoreHiLo1["Case{0}".format(val + 1)] = 0
        StoreOdd2["Case{0}".format(val + 1)] = 0
        StoreHiLo2["Case{0}".format(val + 1)] = 0

    for i in (8, 9, 10, 19, 20, 21, 30, 31, 32, 41, 42, 43, 52, 53, 54):
        soup2 = soup.find(id="main2").find_all("td")[i]
        string1 = str(soup2)
        string1 = string1.replace('<td width="8%">', "")
        string1 = string1.replace('<br/><span class="up">', " ")
        string1 = string1.replace('<br/><span class="down">', " ")
        string1 = string1.replace('<br/><span class="">', " ")
        string1 = string1.replace("</span></td>", "")
        string1 = string1.split()
        match i:
            case 8:
                StoreNumHome["Case1"] = float(string1[0])
                StoreNumHomeAfter["Case1"] = float(string1[1])
            case 9:
                StoreNumDraw["Case1"] = float(string1[0])
                StoreNumDrawAfter["Case1"] = float(string1[1])
            case 10:
                StoreNumAway["Case1"] = float(string1[0])
                StoreNumAwayAfter["Case1"] = float(string1[1])

            case 19:
                StoreNumHome["Case2"] = float(string1[0])
                StoreNumHomeAfter["Case2"] = float(string1[1])
            case 20:
                StoreNumDraw["Case2"] = float(string1[0])
                StoreNumDrawAfter["Case2"] = float(string1[1])
            case 21:
                StoreNumAway["Case2"] = float(string1[0])
                StoreNumAwayAfter["Case2"] = float(string1[1])

            case 30:
                StoreNumHome["Case3"] = float(string1[0])
                StoreNumHomeAfter["Case3"] = float(string1[1])
            case 31:
                StoreNumDraw["Case3"] = float(string1[0])
                StoreNumDrawAfter["Case3"] = float(string1[1])
            case 32:
                StoreNumAway["Case3"] = float(string1[0])
                StoreNumAwayAfter["Case3"] = float(string1[1])

            case 41:
                StoreNumHome["Case4"] = float(string1[0])
                StoreNumHomeAfter["Case4"] = float(string1[1])
            case 42:
                StoreNumDraw["Case4"] = float(string1[0])
                StoreNumDrawAfter["Case4"] = float(string1[1])
            case 43:
                StoreNumAway["Case4"] = float(string1[0])
                StoreNumAwayAfter["Case4"] = float(string1[1])

            case 52:
                StoreNumHome["Case5"] = float(string1[0])
                StoreNumHomeAfter["Case5"] = float(string1[1])
            case 53:
                StoreNumDraw["Case5"] = float(string1[0])
                StoreNumDrawAfter["Case5"] = float(string1[1])
            case 54:
                StoreNumAway["Case5"] = float(string1[0])
                StoreNumAwayAfter["Case5"] = float(string1[1])

    soup4 = soup.find(id="main2").find_all("tr")[0]
    string2 = str(soup4)
    string2 = string2.replace("", "")
    string2 = string2.replace('<font color="yellow">', "")
    string2 = string2.replace("</font>", "")
    string2 = string2.replace("</td>", "")
    string2 = string2.replace(" ", "")
    string2 = string2.replace('<td colspan="13" height="25">', "")
    string2 = string2.replace('<font color="yellow">', "")
    string2 = string2.replace("</font>", "")
    string2 = string2.replace("</td>", "")
    string2 = string2.replace("</tr>", "")
    string2 = string2.replace(" ", "")
    string2 = string2.split()
    gg = string2[0].replace(
        '<tralign="center"class="scoretitle"><tdcolspan="13"height="25">', ""
    )
    list1 = string2[2] + " " + string2[3] + " " + string2[4]

    for i in (13, 16, 24, 27, 35, 38, 46, 49, 57, 60):
        #
        soup4 = soup.find(id="main2").find_all("td")[i]
        # ก้อนใหญ๋
        string3 = str(soup4)
        # ก้อนเล็ก
        soup5 = soup4.find("span")
        string4 = str(soup5)

        oddbefore = string3.replace("{}".format(soup5), "")
        oddbefore = oddbefore.replace('<td width="8%">', "")
        oddbefore = oddbefore.replace("<br/>", "")
        oddbefore = oddbefore.replace("</td>", "")
        find1 = oddbefore.find("/")
        if find1 < 0:
            numbefore = float(oddbefore)
        if find1 > 0:
            oddbefore = oddbefore.replace("/", " ")
            oddbefore = oddbefore.split()
            numbefore = (float(oddbefore[0]) + float(oddbefore[1])) / 2

        oddafter = string4.replace('<span class="up">', "")
        oddafter = oddafter.replace('<span class="">', "")
        oddafter = oddafter.replace('<span class="down">', "")
        oddafter = oddafter.replace("</span>", "")
        oddafter = oddafter.replace("</td>", "")
        find2 = oddafter.find("/")
        if find2 < 0:
            numafter = float(oddafter)
        if find2 > 0:
            oddafter = oddafter.replace("/", " ")
            oddafter = oddafter.split()
            numafter = (float(oddafter[0]) + float(oddafter[1])) / 2

        match i:
            case 13:
                StoreOdd1["Case1"] = numbefore
                StoreOdd2["Case1"] = numafter
            case 16:
                StoreHiLo1["Case1"] = numbefore
                StoreHiLo2["Case1"] = numafter

            case 24:
                StoreOdd1["Case2"] = numbefore
                StoreOdd2["Case2"] = numafter
            case 27:
                StoreHiLo1["Case2"] = numbefore
                StoreHiLo2["Case2"] = numafter

            case 35:
                StoreOdd1["Case3"] = numbefore
                StoreOdd2["Case3"] = numafter
            case 38:
                StoreHiLo1["Case3"] = numbefore
                StoreHiLo2["Case3"] = numafter

            case 46:
                StoreOdd1["Case4"] = numbefore
                StoreOdd2["Case4"] = numafter
            case 49:
                StoreHiLo1["Case4"] = numbefore
                StoreHiLo2["Case4"] = numafter

            case 57:
                StoreOdd1["Case5"] = numbefore
                StoreOdd2["Case5"] = numafter
            case 60:
                StoreHiLo1["Case5"] = numbefore
                StoreHiLo2["Case5"] = numafter

    oddbeforeavg = (
        StoreOdd1["Case1"]
        + StoreOdd1["Case2"]
        + StoreOdd1["Case3"]
        + StoreOdd1["Case4"]
        + StoreOdd1["Case5"]
    ) / 5
    oddafteravg = (
        StoreOdd2["Case1"]
        + StoreOdd2["Case2"]
        + StoreOdd2["Case3"]
        + StoreOdd2["Case4"]
        + StoreOdd2["Case5"]
    ) / 5
    hilobeforeavg = (
        StoreHiLo1["Case1"]
        + StoreHiLo1["Case2"]
        + StoreHiLo1["Case3"]
        + StoreHiLo1["Case4"]
        + StoreHiLo1["Case5"]
    ) / 5
    hiloaftereavg = (
        StoreHiLo2["Case1"]
        + StoreHiLo2["Case2"]
        + StoreHiLo2["Case3"]
        + StoreHiLo2["Case4"]
        + StoreHiLo2["Case5"]
    ) / 5

    homeavg = (
        StoreNumHome["Case1"]
        + StoreNumHome["Case2"]
        + StoreNumHome["Case3"]
        + StoreNumHome["Case4"]
        + StoreNumHome["Case5"]
    ) / 5
    awayavg = (
        StoreNumAway["Case1"]
        + StoreNumAway["Case2"]
        + StoreNumAway["Case3"]
        + StoreNumAway["Case4"]
        + StoreNumAway["Case5"]
    ) / 5
    drawavg = (
        StoreNumDraw["Case1"]
        + StoreNumDraw["Case2"]
        + StoreNumDraw["Case3"]
        + StoreNumDraw["Case4"]
        + StoreNumDraw["Case5"]
    ) / 5
    homeavgAfter = (
        StoreNumHomeAfter["Case1"]
        + StoreNumHomeAfter["Case2"]
        + StoreNumHomeAfter["Case3"]
        + StoreNumHomeAfter["Case4"]
        + StoreNumHomeAfter["Case5"]
    ) / 5
    awayavgAfter = (
        StoreNumAwayAfter["Case1"]
        + StoreNumAwayAfter["Case2"]
        + StoreNumAwayAfter["Case3"]
        + StoreNumAwayAfter["Case4"]
        + StoreNumAwayAfter["Case5"]
    ) / 5
    drawavgAfter = (
        StoreNumDrawAfter["Case1"]
        + StoreNumDrawAfter["Case2"]
        + StoreNumDrawAfter["Case3"]
        + StoreNumDrawAfter["Case4"]
        + StoreNumDrawAfter["Case5"]
    ) / 5

    print(
        "%.3f" % homeavg,
        ",",
        "%.3f" % awayavg,
        ",",
        "%.3f" % drawavg,
        ",",
        "%.3f" % homeavgAfter,
        ",",
        "%.3f" % awayavgAfter,
        ",",
        "%.3f" % drawavgAfter,
        ",",
        "%.3f" % (homeavg - homeavgAfter),
        ",",
        "%.3f" % (awayavg - awayavgAfter),
        ",",
        "%.3f" % (drawavg - drawavgAfter),
        ",",
        oddbeforeavg,
        ",",
        oddafteravg,
        ",",
        "%.3f" % (oddbeforeavg - oddafteravg),
        ",",
        hilobeforeavg,
        ",",
        hiloaftereavg,
        ",",
        "%.3f" % (hilobeforeavg - hiloaftereavg),
        ",",
        "",
        list1,
        gg,
    )
    list2 = [
        [
            "%.3f" % homeavg,
            "%.3f" % awayavg,
            "%.3f" % drawavg,
            "%.3f" % homeavgAfter,
            "%.3f" % awayavgAfter,
            "%.3f" % drawavgAfter,
            "%.3f" % (homeavg - homeavgAfter),
            "%.3f" % (awayavg - awayavgAfter),
            "%.3f" % (drawavg - drawavgAfter),
            oddbeforeavg,
            oddafteravg,
            "%.3f" % (oddbeforeavg - oddafteravg),
            hilobeforeavg,
            hiloaftereavg,
            "%.3f" % (hilobeforeavg - hiloaftereavg),
            "",
            list1,
            gg,
        ]
    ]
    list3 = [
        "%.3f" % homeavg,
        "%.3f" % awayavg,
        "%.3f" % drawavg,
        "%.3f" % homeavgAfter,
        "%.3f" % awayavgAfter,
        "%.3f" % drawavgAfter,
        "%.3f" % (homeavg - homeavgAfter),
        "%.3f" % (awayavg - awayavgAfter),
        "%.3f" % (drawavg - drawavgAfter),
        oddbeforeavg,
        oddafteravg,
        "%.3f" % (oddbeforeavg - oddafteravg),
        hilobeforeavg,
        hiloaftereavg,
        "%.3f" % (hilobeforeavg - hiloaftereavg),
    ]
    b = np.array(list3, dtype=float)
    nameteam = list1 + gg

    def write_to_csv(list_of_emails):
        with open("data2.csv", "a", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            writer.writerows(list_of_emails)

    write_to_csv(list2)
    pass


@app.route("/callPredict")
def callPredict():
    # ใส่โค้ด callPredict ของคุณที่นี่global nameteam
    global b
    dataset = pd.read_csv(r"newmodel.csv")
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    dataset = pd.read_csv(r"newmodel.csv")
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1, random_state=0
    )

    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    from sklearn.tree import DecisionTreeClassifier

    classifier = DecisionTreeClassifier(criterion="entropy", random_state=0)
    classifier.fit(X_train, y_train)

    from sklearn.neighbors import KNeighborsClassifier

    classifier1 = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
    classifier1.fit(X_train, y_train)

    from sklearn.svm import SVC

    classifier2 = SVC(kernel="rbf", random_state=0)
    classifier2.fit(X_train, y_train)

    from sklearn.linear_model import LogisticRegression

    classifier3 = LogisticRegression(random_state=0)
    classifier3.fit(X_train, y_train)

    from sklearn.naive_bayes import GaussianNB

    classifier4 = GaussianNB()
    classifier4.fit(X_train, y_train)

    from sklearn.ensemble import RandomForestClassifier

    classifier5 = RandomForestClassifier(
        n_estimators=10, criterion="entropy", random_state=0
    )
    classifier5.fit(X_train, y_train)

    from sklearn.svm import SVC

    classifier6 = SVC(kernel="linear", random_state=0)
    classifier6.fit(X_train, y_train)

    from sklearn.svm import SVC

    classifier7 = SVC(kernel="poly", random_state=0)
    classifier7.fit(X_train, y_train)

    from sklearn.svm import SVC

    classifier8 = SVC(kernel="rbf", random_state=0)
    classifier8.fit(X_train, y_train)

    from sklearn.svm import SVC

    classifier9 = SVC(kernel="sigmoid", random_state=0)
    classifier9.fit(X_train, y_train)

    a = [b]

    y_pred = classifier.predict(a)
    y_pred2 = classifier1.predict(a)
    y_pred4 = classifier2.predict(a)
    y_pred6 = classifier3.predict(a)
    y_pred8 = classifier4.predict(a)
    y_pred10 = classifier5.predict(a)
    y_pred12 = classifier6.predict(a)
    y_pred14 = classifier7.predict(a)
    y_pred16 = classifier8.predict(a)
    y_pred18 = classifier9.predict(a)
    x1 = int(y_pred)
    x2 = int(y_pred2)
    x3 = int(y_pred4)
    x4 = int(y_pred6)
    x5 = int(y_pred8)
    x6 = int(y_pred10)
    x7 = int(y_pred12)
    x8 = int(y_pred14)
    x9 = int(y_pred16)
    x10 = int(y_pred18)
    testgroupby = [[x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, 3]]

    def write_to_csv(list_of_emails):
        with open("testgroupby.csv", "a", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            writer.writerows(list_of_emails)

    write_to_csv(testgroupby)
    pass


@app.route("/callResult")
def callResult():
    # ใส่โค้ด callResult ของคุณที่นี่
    global nameteam
    dataset = pd.read_csv(r"testgroupby.csv")
    p = dataset.iloc[:, -1].values
    item_group = dataset.groupby(
        [
            "group1",
            "group2",
            "group3",
            "group4",
            "group5",
            "group6",
            "group7",
            "group8",
            "group9",
            "group10",
        ]
    )
    item_group.groups
    y2 = dataset.groupby(
        [
            "group1",
            "group2",
            "group3",
            "group4",
            "group5",
            "group6",
            "group7",
            "group8",
            "group9",
            "group10",
        ]
    ).count()
    allCase = y2["result"].values
    storeCase2 = []
    Store2 = {}
    resultforall2 = {}

    test = []
    test2 = []
    storeCorrect = {}
    storeInCorrect = {}
    print(y2)
    for group1, group2 in item_group:
        group1 = list(group1)
        storeCase2.append(group1)
    # print(len(storeCase))
    checkcaseandstore = {}
    for val in range(len(storeCase2)):
        Store2["Case{0}".format(val + 1)] = storeCase2[val]
        checkcaseandstore["Case{0}".format(val + 1)] = 0
        resultforall2["Case{0}".format(val + 1)] = """have {0} case""".format(
            allCase[val]
        )
        storeCorrect["Case{0}".format(val + 1)] = 0
        storeInCorrect["Case{0}".format(val + 1)] = 0
    # print(Store2)
    for g in range(len(p)):
        global testcode
        allpredict = dataset.iloc[g : g + 1, 0:10].values
        # print(z)
        result = dataset.iloc[g : g + 1, -1].values
        test.append(allpredict.tolist())
        test2.append(result.tolist())
        for val in range(len(storeCase2)):
            if test[g] == [Store2["Case{0}".format(val + 1)]]:
                checkcaseandstore["Case{0}".format(val + 1)] = (
                    checkcaseandstore["Case{0}".format(val + 1)] + 1
                )
                if test2[g] == [1]:
                    storeCorrect["Case{0}".format(val + 1)] = (
                        storeCorrect["Case{0}".format(val + 1)] + 1
                    )
                if test2[g] == [0]:
                    storeInCorrect["Case{0}".format(val + 1)] = (
                        storeInCorrect["Case{0}".format(val + 1)] + 1
                    )
                if test2[g] == [3]:
                    all = checkcaseandstore["Case{0}".format(val + 1)] - 1
                    home = storeCorrect["Case{0}".format(val + 1)]
                    away = storeInCorrect["Case{0}".format(val + 1)]
                    print(home)
                    print(away)
                    if home > away:
                        #     print("Case {0} have".format(val+1),all,"Home",home,"Away",away,"Percent Home",((home)/all)*100,"%")
                        testcode = (
                            """{} Case {} have {} home {} away {} Percent Home {} %""".format(
                                nameteam,
                                val + 1,
                                all,
                                home,
                                away,
                                (home / (home + away)) * 100,
                            )
                            + "\n"
                        )
                    if home < away:
                        #     print("Case {0} have".format(val+1),all,"Home",home,"Away",away,"Percent Away",((away)/all)*100,"%")
                        testcode = (
                            """{} Case {} have {} home {} away {} Percent Away {} %""".format(
                                nameteam,
                                val + 1,
                                all,
                                home,
                                away,
                                (away / (home + away)) * 100,
                            )
                            + "\n"
                        )
                    else:
                        testcode = "Null"
                pass


if __name__ == "__main__":
    app.run(debug=True)
