import tweepy
from project_template import qf
import os.path
import io

CONSUMER_KEY =          "hgxRhymigitU3bwHVf8jnNcP2"
CONSUMER_SECRET =       "JcFNhLuiBtSqdvjKec1pwLiZyDzrY9IrlTJd8DEKjAO7qqkE9k"
ACCESS_TOKEN =          "726989412953952256-1hvT57RxiLm3egUbagMumUvYi5mO5XV"
ACCESS_TOKEN_SECRET =   "rZUR7x4VkliT7TKMfdxfbyyBO6rdudy0J3W7BxT6XrVwz"

def get_tweet():
	auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
	auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
	api = tweepy.API(auth)

	seen_tweets = []
	with open("jsons/seen_tweets.txt") as f:
		seen_tweets = f.readlines()
		seen_tweets = [line.strip() for line in seen_tweets]

	twts = api.search(q="@annoying_chat")     
	for s in twts:
		text =  str(s.text)
		if "ANC" in text and not (s.user.screen_name == "annoying_chat"):
			text = text[(text.index("ANC") + len("ANC: ")):]
			sn = s.user.screen_name
			skip_tweet = False		
			for t in seen_tweets:
				if text + sn == t:
					skip_tweet = True
					print "Skipping tweet: %s" % text
			if skip_tweet: continue
			response = qf.find_final(text)[0][0]
			m = "@%s %s" % (sn, response)
			m = m[:140] if len(m) > 140 else m
			try:				
				s = api.update_status(m, s.id)
				print "Found new tweet to reply to:\n%s" % text
				seen_tweets.append(text + sn)
			except:
				pass
				#do nothing if status is a duplicate

	with open("jsons/seen_tweets.txt", 'w') as f:
		for t in seen_tweets:
			f.write(str(t) + "\n")

	return "Processed Tweets."

#Tried looking at random tweets from an incoming stream, but many were in various languages and/or did not fit the "chat" style.