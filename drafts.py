# BUILDING TEST SET
def buildTestSet(search_keyword):
    try:
        tweets_fetched = twitter_API.GetSearch(search_keyword, count = 20)
        dataLine = [{"text":status.text, "label":None} for status in tweets_fetched]
        with open(testDataFile,'w', encoding="utf-8") as csvfile:
            linewriter = csv.writer(csvfile,delimiter=',',quotechar="\"")
            try:
                linewriter.writerow({"text":status.text, "label":None} for status in tweets_fetched)
            except Exception as e:
                print(e)
            print("Fetched " + str(len(tweets_fetched)) + " tweets for the term " + search_keyword)
        return dataLine
    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print (message)
        return None

testDataFile = "testDataFile.csv"
testDataSet = buildTestSet("apple")
