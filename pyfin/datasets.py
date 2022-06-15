# import yfinance as yf
# from yahoofinancials import YahooFinancials
import json
import numpy as np


def TSLA():

	with open("../data/TSLA.json", "r") as fp:
		data = json.load(fp)
		data["dt"] = 1

	S = np.array(data["S"])
	t = np.array(data["dates"]) + 693960  # days since year 0000
	dt = data["dt"]

	return t, S, dt


def download_dataset(tag: str = "", period: str = "5y"):
	if tag:
		pass
		# ticker = yf.Ticker(tag)
		# tag_df = ticker.history(period="5y")
		# data = tag_df["Close"]

	# S = data["S"]
	# t = data["dates"]
	# dt = data["dt"]
	# return t, S, dt
