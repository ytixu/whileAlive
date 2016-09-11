import csv
import json
import requests
import sys
import warnings

warnings.filterwarnings('ignore')

server = 'https://rest.ensembl.org/taxonomy/classification/'
headers= { 'Content-Type' : 'application/json' }
data_folder = '../data/ensembl/'

class Taxonomy:

	def __init__(self):
		self._id_number = 0
		self._complete_id_list = {}
		self._id_list = [2759] # eucaryotes

	def _isNotInList_or_addToList(self, species_id):
		if species_id not in self._complete_id_list:
			self._id_list.append(species_id)
			return True

		return False

	def getRequest(self):
		endpoint = server + str(self._id_number) + '?'
		r = requests.get(endpoint, headers=headers, verify=False)

		if not r.ok:
		  r.raise_for_status()
		  sys.exit()

		decoded = r.json()
		# print repr(decoded)
		return decoded


	def getData(self, data_file_id):
		data = self.getRequest()
		new_ids = []
		output = {}

		with open('../data/ensembl/taxonomy.csv', 'a') as csv_file:
			csv_writer = csv.writer(csv_file, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)

			for species in data:
				if species['id'] in self._complete_id_list:
					continue

				output[species['id']] = species

				#indicate that it's done
				self._complete_id_list[species['id']] = True
				# add new ids
				children_id = []
				if 'children' in species:
					children_id = map(lambda x: x['id'], species['children'])

				new_ids += children_id + [species['parent']['id']]
				# white to csv
				write_data = [species['id'], species['scientific_name'], species['parent']['id'], data_file_id]
				csv_writer.writerow(write_data)

		for species_id in set(new_ids):
			self._isNotInList_or_addToList(species_id)

		# print repr(self._id_list)

		self._complete_id_list[self._id_number] = True
		return output

	def run(self):
		with open(data_folder+'taxonomy.csv', 'wb') as csv_file:
			csv_writer = csv.writer(csv_file, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
			csv_writer.writerow(['ID', 'NAME', 'PARENT', 'FILE'])

		file_index = 0
		data = {}
		while self._id_list:
			self._id_number = self._id_list.pop()

			print self._id_number

			result = self.getData(file_index)
			data.update(result)

			if len(data) >= 10:
				with open(data_folder+'data_'+str(file_index)+'.json', 'w') as json_file:
					json.dump(data, json_file)

				file_index += 1
				data = {}


if __name__ == "__main__":
	txy = Taxonomy()
	txy.run()