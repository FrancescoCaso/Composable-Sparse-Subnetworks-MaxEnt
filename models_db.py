from pathlib import Path
import yaml
from pymongo import MongoClient
from tinydb import TinyDB, Query, table
from pymongo.server_api import ServerApi

from train_utils import ClassesList


result_pths = ['results', 'lth_results']

class ModelsDB:
    def __init__(self, 
                 mongo_uri="",
                 database_path=Path(__file__).parent / "database.json",
                 use_mongo=False,
                 db_name="ModelsDB",
                 collection_name="models",
                 refresh=False,
                 result_paths=result_pths):

        self.paths = result_paths
        self.use_mongo = use_mongo
        if use_mongo:
            self.client = MongoClient(mongo_uri, server_api=ServerApi('1'))
            try:
                self.client.admin.command('ping')
                print("Pinged your deployment. You successfully connected to MongoDB!")
            except Exception as e:
                print(e)

            self.db = self.client[db_name]
        
            self.collection = self.db[collection_name]
            if refresh:
                self.db.drop_collection(collection_name)
                print(f"Collection {collection_name} dropped.")
                self.instantiate_database()
        else:
            self.Models = Query()
            self.database_path = database_path
            self.instantiate_tinydb(refresh=refresh)

        

    def remove_from_conf(self, conf):
        conf['model_path'] = Path(conf['model_path']).unlink()
        conf['model_path'] = Path(conf['model_path']).with_name(f"{conf['file_id']}_config.yaml").unlink()
        self.collection.delete_one({"file_id": conf['file_id']})

    def instantiate_tinydb(self, refresh=True):
        """
        Instantiate a TinyDB database to store model configurations and results.
        
        Args:
            refresh (bool): If True, refresh the database by deleting existing entries.
        """
        if refresh:
            self.database_path.unlink(missing_ok=True)
        
        self.db = TinyDB(self.database_path)
        for result_pth in self.paths:
            for conf_file in Path(result_pth).rglob("*.yaml"):
                if "_raw" in conf_file.stem:
                    continue
                file_id = conf_file.stem.split('_config')[0] # Extract the file ID from the path
                if any(self.db.search(self.Models.file_id == file_id)):
                    continue
                with open(conf_file, "r") as f:
                    data = yaml.safe_load(f)
                    data['model_path'] = (conf_file.parent / f"{file_id}.pt").as_posix()

                    data['file_id'] = file_id
                    assert Path(data['model_path']).exists(), f"Model file {data['model_path']} does not exist."
                    data['rewarded_classes'] = [i for i, w in enumerate(data['weights']) if w == 1]

                    doc_id = int(file_id, 16) % (10 ** 8)
                    while self.db.contains(doc_id=doc_id):
                        doc_id = (doc_id + 1) % (10 ** 8)
                    self.db.insert(table.Document(data, doc_id=doc_id))
        
    def instantiate_database(self):
        for result_pth in self.paths:
            for conf_file in Path(result_pth).rglob("*.yaml"):
                if "_raw" in conf_file.stem:
                    continue
                file_id = conf_file.stem.split('_config')[0] # Extract the file ID from the path
                if self.collection.find_one({"file_id": file_id}):
                    continue
                with open(conf_file, "r") as f:
                    data = yaml.safe_load(f)

                    data['model_path'] = (conf_file.parent / f"{file_id}.pt").as_posix()
                    data['file_id'] = file_id
                    assert Path(data['model_path']).exists(), f"Model file {data['model_path']} does not exist."
                    data['rewarded_classes'] = [i for i, w in enumerate(data['weights']) if w == 1]
                    self.collection.insert_one(data)

    def add(self, config, hash):
        """
        Add a new configuration to the database.
        
        Args:
            config (dict): The configuration to add.
        """
        config['rewarded_classes'] = [i for i, w in enumerate(config['weights']) if w == 1]
        config['file_id'] = hash
        if isinstance(config['model_path'], Path):
            config['model_path'] = config['model_path'].as_posix()
        if isinstance(config['classes'], ClassesList):
            config['classes'] = config['classes'].classes
        
        if self.use_mongo:
            if not self.collection.find_one({"file_id": hash}):
                self.collection.insert_one(config)
            else:
                raise ValueError(f"Configuration with file_id {hash} already exists in the database.")
        else:
            doc_id = int(hash, 16) % (10 ** 8)
            while self.db.contains(doc_id=doc_id):
                doc_id = (doc_id + 1) % (10 ** 8)

            self.db.insert(table.Document(config, doc_id=doc_id))
        
    def query(self, config, seed=None, rewarded_class=None, dataset_name=None):
        if not self.use_mongo:
            return self.tinydb_query(config, seed=seed, rewarded_class=rewarded_class, dataset_name=dataset_name)
        rewarded_class = [rewarded_class] if isinstance(rewarded_class, int) else rewarded_class
        if dataset_name is None:
            dataset_name = config['dataset_name'] # enable multidataset querying

        q = {
            "model.name": config["model"]["name"],
            "dataset_name": dataset_name,
            "lth": config["lth"],
        }
        if rewarded_class:
            q["rewarded_classes"] = list(rewarded_class)
        if seed is not None:
            q["seed"] = seed

        if config["model"]["name"] == 'MLP':
            q['model.hidden_dims'] = config['model']['hidden_dims']
            q['model.dropout'] = config['model']['dropout']
            q['model.t'] = config["model"]["t"]
            q['optimizer.lr'] = config['optimizer']['lr']
            # TODO: aggiungere altri campi se necessario
        elif config["model"]["name"] == 'ConvNet':
            q['model.hidden_conv_dims'] = config['model']['hidden_conv_dims']
            q['model.pool_kernel_size'] = config['model']['pool_kernel_size']
            q['model.pool_stride'] = config['model']['pool_stride']
            q['model.t'] = config["model"]["t"]
        else:
            raise NotImplementedError(f"Model {config['model']['name']} not implemented.")
        
        if config["lth"]:
            q['prune_ratio'] = config['prune_ratio']
            q['prune_iter'] = config['prune_iter']
        
        found = self.collection.find(q)
        return list(found)
    
    def tinydb_query(self, config, seed=None, rewarded_class=None, dataset_name=None):
        rewarded_class = [rewarded_class] if isinstance(rewarded_class, int) else rewarded_class
        if dataset_name is None:
            dataset_name = config['dataset_name'] # enable multidataset querying
        # assert max(rewarded_class) < len(config["classes"]), f"Rewarded class {rewarded_class} is out of bounds for the number of classes {len(config['classes'])}."
        if config["model"]["name"] == 'MLP':
            results = self.db.search(
                # (self.Models.seed == seed) &
                (self.Models.model.name == config['model']['name']) &
                (self.Models.rewarded_classes.test(lambda c: list(c) == list(rewarded_class))) &
                (self.Models.dataset_name == dataset_name) &
                (self.Models.lth == config['lth']) &
                (self.Models.model.hidden_dims == config['model']['hidden_dims']) &
                (self.Models.model.dropout == config['model']['dropout']) &
                (self.Models.optimizer.lr == config['optimizer']['lr']) &
                # (self.Models.epochs == config['epochs']) &
                (self.Models.model.t == config["model"]["t"])
                # (Models.optimizer.betas == config['optimizer']['betas'])
                # TODO: aggiungere altri campi se necessario
            )
        elif config["model"]["name"] == 'ConvNet':
            results = self.db.search(
                (self.Models.model.name == config['model']['name']) &
                # (self.Models.seed == seed) &
                (self.Models.rewarded_classes.test(lambda c: list(c) == list(rewarded_class))) &
                (self.Models.dataset_name == dataset_name) &
                (self.Models.lth == config['lth']) &
                (self.Models.model.t == config["model"]["t"]) &
                (self.Models.model.pool_kernel_size == config['model']['pool_kernel_size']) &
                (self.Models.model.pool_stride == config['model']['pool_stride']) &
                (self.Models.model.hidden_conv_dims == config['model']['hidden_conv_dims'])
            )   
        else:
            raise NotImplementedError(f"Model {config['model']['name']} not implemented.")
        
        if config["lth"]:
            results = [r for r in results if r['prune_ratio'] == config['prune_ratio'] and r['prune_iter'] == config['prune_iter']]
        if seed is not None:
            results = [r for r in results if r['seed'] == seed]
        return results
    

if __name__ == "__main__":
    db = ModelsDB()

#     results = results = db.db.search(
#                 (db.Models.model.name == 'ConvNet') &
#                 (db.Models.dataset_name == 'mnist') &
#                 (db.Models.model.t == 5)
#             ) 
    
#     for r in results:
#         id = r['file_id']
#         model_path = r['model_path']
#         print(f"removing {id} from database")
        
#         Path(model_path).with_name(f"{id}_config.yaml").unlink()
#         Path(model_path).with_name(f"{id}_config_raw.yaml").unlink(missing_ok=True)
#         Path(model_path).unlink()

#     db = ModelsDB(refresh=True)
    # from tqdm import tqdm
    # db = ModelsDB()
    # for conf in tqdm(list(Path("configs").rglob("setting*mlp*.yaml"))):
    #     with open(conf, "r") as f:
    #         config = yaml.safe_load(f)
    #     for cl in config['classes']:
    #         for seed in range(5):
    #             rs = db.query(config, seed=seed, rewarded_class=cl)
    #             if len(rs) > 1:
    #                 rs = sorted(rs, key=lambda x: x['epochs'], reverse=True)
    #                 for r in rs[1:]:
    #                     # remove the model file and the config file
    #                     uncommons = set()
    #                     for k in rs[0]:
    #                         if k == 'seeds':
    #                             continue
    #                         if rs[0][k] != r[k]:
    #                             uncommons.add(k)
                        
    #                     if 'epochs' in uncommons:
    #                         assert config['epochs'] == rs[0]['epochs']
    #                         uncommons.remove('epochs') # irrelevant
    #                     assert uncommons == {'file_id', 'model_path'}
    #                     if Path(r['model_path']).exists():
    #                         Path(r['model_path']).unlink()
    #                     if Path(r['model_path']).with_name(f"{r['file_id']}_config.yaml").exists():
    #                         Path(r['model_path']).with_name(f"{r['file_id']}_config.yaml").unlink()
    #                 print(conf.stem)
    #                 print(cl, seed, len(rs), '\n')