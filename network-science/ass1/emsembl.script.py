import requests, sys, json
 
server = 'https://rest.ensembl.org/taxonomy/id/'
headers= { 'Content-Type' : 'application/json' }

class Taxonomy:

	def __init__(self):
		self._id_number = 0
		self._id_list = []


	def getRequest(self):
		endpoint = server + self._id_number + '?'
		r = requests.get(endpoint, headers)
	 
		if not r.ok:
		  r.raise_for_status()
		  sys.exit()
		 
		decoded = r.json()
		print repr(decoded)

if __name__ == "__main__":
	txy = Taxonomy()
	txy.getRequest()