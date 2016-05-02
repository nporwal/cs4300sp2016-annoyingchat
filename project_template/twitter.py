import tweepy
from project_template import qf


CONSUMER_KEY =          "hgxRhymigitU3bwHVf8jnNcP2"
CONSUMER_SECRET =       "JcFNhLuiBtSqdvjKec1pwLiZyDzrY9IrlTJd8DEKjAO7qqkE9k"
ACCESS_TOKEN =          "726989412953952256-1hvT57RxiLm3egUbagMumUvYi5mO5XV"
ACCESS_TOKEN_SECRET =   "rZUR7x4VkliT7TKMfdxfbyyBO6rdudy0J3W7BxT6XrVwz"

def get_tweet():
	auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
	auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
	api = tweepy.API(auth)
	 
	twts = api.search(q="@annoying_chat")     
	for s in twts:
		text =  str(s.text)
		if "ANC" in text and not (s.user.screen_name == "annoying_chat"):
			text = text[(text.index("ANC") + len("ANC: ")):]
			sn = s.user.screen_name
			results = qf.find_final(text)
			for res in results:
				response = res[0]
				m = "@%s %s!" % (sn, response)
				if len(m) > 140: continue
			try:				
				s = api.update_status(m, s.id)
				print "Found new tweet to reply to:\n%s" % text
			except:
				pass
				#do nothing if status is a duplicate
	return "Processed Tweets."

'''
consumer_key = "hgxRhymigitU3bwHVf8jnNcP2"
consumer_secret = "JcFNhLuiBtSqdvjKec1pwLiZyDzrY9IrlTJd8DEKjAO7qqkE9k"
access_token = "726989412953952256-1hvT57RxiLm3egUbagMumUvYi5mO5XV"
access_token_secret = "rZUR7x4VkliT7TKMfdxfbyyBO6rdudy0J3W7BxT6XrVwz"

import time

def log_error(msg):
    timestamp = time.strftime('%Y%m%d:%H%M:%S')
    sys.stderr.write("%s: %s\n" % (timestamp,msg))

class StreamWatcherListener(tweepy.StreamListener):

	tweet = ''

	def on_status(self, status):
		print status.text.encode('utf-8')
		with open("jsons/twit.txt", 'w') as f:
			f.write(status.text.encode('utf-8'))
		return False

	def on_error(self, status_code):
		log_error("Status code: %s." % status_code)
		time.sleep(3)
		return True  # keep stream alive

	def on_timeout(self):
		log_error("Timeout.")


def get_tweet():
	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_token_secret)
	listener = StreamWatcherListener()
	stream = tweepy.Stream(auth, listener)
	stream.sample()
	with open("jsons/twit.txt") as f:
		tweet = f.readlines()[0]
		output_list = qf.find_final(tweet)
	return output_list, tweet
'''

'''
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
public_tweets = api.home_timeline()
return [tweet.text for tweet in public_tweets]
'''