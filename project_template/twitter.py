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
			response = qf.find_final(text)[0][0]
			m = "@%s %s" % (sn, response)
			m = m[:140] if len(m) > 140 else m
			try:				
				s = api.update_status(m, s.id)
				print "Found new tweet to reply to:\n%s" % text
			except:
				pass
				#do nothing if status is a duplicate
	return "Processed Tweets."

#Tried looking at random tweets from an incoming stream, but many were in various languages and/or did not fit the "chat" style.