import Add from "@mui/icons-material/Add";
import Button from "@mui/material/Button"
import Card from "@mui/material/Card"
import CardContent from "@mui/material/CardContent"
import Typography from "@mui/material/Typography"
import List from "@mui/material/List"
import ListItem from "@mui/material/ListItem"
import { useState, useEffect } from "react";
import { v4 as uuidv4 } from "uuid";
import {
  LayerConfig,
  SetLocalStorageEntity,
} from "../Utility/DefaultConfigs";
import Box from "@mui/material/Box/Box";
import Layer from "./Layer";

interface SampleProps {
  jsonCallback: any;
}

const Sample = (props: SampleProps) => {
  const [jsonData, setJsonData] = useState({});
  const [layers, setLayers] = useState<any>([]);

  const localStorageEntityName: string = "sampleConfig";
  const configFieldName: string = "sample";

  useEffect(() => {

    let formattedShapes = Object.keys(jsonData).map((key) => jsonData[key]);
    if (formattedShapes.length == 0) { return; }

    let substrate = formattedShapes[0]
    delete substrate["thickness"];

    let formattedLayers = formattedShapes.slice(1)
    let layersWithOrder = formattedLayers.map((layer, i) => { layer["order"] = i; return layer; })
    let json = {
      "substrate": substrate,
      "layers": layersWithOrder
    }

    props.jsonCallback(json, configFieldName);

    SetLocalStorageEntity(formattedShapes, [], localStorageEntityName);
  }, [jsonData]);

  useEffect(() => {
    let data = localStorage.getItem(localStorageEntityName);
    if (data !== null) {
      let sampleConfig = JSON.parse(data);

      let substrate = sampleConfig[0]
      let cachedLayers: any = [<Layer
        key={"0"}
        id={"0"}
        initialConfig={substrate}
        jsonCallback={createJsonForLayer}
      />];
      let otherLayers = sampleConfig.slice(1)

      for (var layer of otherLayers) {
        cachedLayers = [...cachedLayers, createLayer(layer)];
      }

      setLayers(cachedLayers);
    }
    else {
      setLayers([<Layer
        key={"0"}
        id={"0"}
        initialConfig={LayerConfig}
        jsonCallback={createJsonForLayer}
      />])
    }
  }, []);

  const removeLayer = (id: string) => {
    setLayers((layers) => layers.filter((layer) => layer.props.id !== id));
    setJsonData((jsonData) => {
      delete jsonData[id];
      return { ...jsonData };
    });
  };

  const createJsonForLayer = (layerJson, layerId) => {
    setJsonData((jsonData) => {
      jsonData[layerId] = layerJson;
      return { ...jsonData };
    });
  };

  const createLayer = (layerConfig) => {
    const myid = uuidv4();
    return (
      <Layer
        key={myid}
        id={myid}
        removeCallback={() => removeLayer(myid)}
        initialConfig={layerConfig}
        jsonCallback={createJsonForLayer}
      />
    );
  };

  const addLayer = () => {
    setLayers([...layers, createLayer(LayerConfig)]);
  };

  return (
      <Card sx={{ height: "100%" }}>
        <CardContent sx={{ height: "100%" }}>
          <Box display="flex" sx={{ flexDirection: "column", height: "100%" }}>
            <Box display="flex" justifyContent={"space-between"} sx={{ paddingBottom: 1 }}>
              <Typography>Sample</Typography>
              <Button size="small"  sx={{color: "text.primary"}} onClick={addLayer}>
                <Add />
              </Button>
            </Box >

            <List sx={{ height: "100%", overflow: "auto" }}>
              {layers.map((value) => {
                return <ListItem key={value.props.id}>{value}</ListItem>;
              })}
            </List>
          </Box>
        </CardContent>
      </Card>
  );
};

export default Sample;
