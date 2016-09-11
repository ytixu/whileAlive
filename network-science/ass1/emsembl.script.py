import requests, sys, json

server = 'https://rest.ensembl.org/taxonomy/classification/'
headers= { 'Content-Type' : 'application/json' }

class Taxonomy:

	def __init__(self):
		self._id_number = 9606
		self._complete_id_list = {}
		self._id_list = []

	def getRequest(self):
		endpoint = server + str(self._id_number) + '?'
		r = requests.get(endpoint, headers=headers, verify=False)

		if not r.ok:
		  r.raise_for_status()
		  sys.exit()

		decoded = r.json()
		# print repr(decoded)
		return decoded

	def getData(self):
		data = self.getRequest()
		output = {}

		for species in data:
			if species['id'] not in self._complete_id_list:
				self._complete_id_list[species['id']] = True
				# reduce child and parent to ids
				output[species['id']] = species.copy()
				output[species['id']]['parent'] = species['parent']['id']
				output[species['id']]['children'] = map(lambda x: x['id'], species['children'])

		print repr(output)



if __name__ == "__main__":
	txy = Taxonomy()
	txy.getData()