import Add from "@mui/icons-material/Add";
import Button from "@mui/material/Button"
import Card from "@mui/material/Card"
import CardActions from "@mui/material/CardActions"
import CardContent from "@mui/material/CardContent"
import MenuItem from "@mui/material/MenuItem"
import Menu from "@mui/material/Menu"
import Grid from "@mui/material/Grid"
import Typography from "@mui/material/Typography"
import List from "@mui/material/List"
import ListItem from "@mui/material/ListItem"
import React, { useEffect } from "react";
import { v4 as uuidv4 } from "uuid";
import {
  LayerConfig,
  SampleConfig,
  SetLocalStorageEntity,
} from "../Utility/DefaultConfigs";
import Box from "@mui/material/Box/Box";
import Layer from "./Layer";

interface SampleProps {
  jsonCallback: any;
}

const Sample = (props: SampleProps) => {
  const [jsonData, setJsonData] = React.useState({});
  const [layers, setLayers] = React.useState<any>([]);



  const localStorageEntityName: string = "sampleConfig2";
  const configFieldName: string = "sample";

  useEffect(() => {
    let formattedShapes = Object.keys(jsonData).map((key) => jsonData[key]);
    let substrate = formattedShapes[0]
    // delete substrate["thickness"];

    let formattedLayers = formattedShapes.slice(1)
    let json = {
      "substrate": substrate,
      "layers": formattedLayers
    }

    props.jsonCallback(json, configFieldName);

    SetLocalStorageEntity(formattedShapes, [], localStorageEntityName);
  }, [jsonData]);

  useEffect(() => {
    setLayers([<Layer
      key={"0"}
      id={"0"}
      initialConfig={LayerConfig}
      jsonCallback={createJsonForLayer}
    />])

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
    <Card style={{ maxHeight: 700, overflow: "auto" }}>
      <CardContent >
        <Box display="flex" sx={{ flexDirection: "column" }}>
          <Box display="flex" justifyContent={"space-between"} sx={{ paddingBottom: 1 }}>
            <Typography>GisaxsShapesConfig</Typography>
            <Button size="small" onClick={addLayer}>
              <Add />
            </Button>
          </Box >

          <List>
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
