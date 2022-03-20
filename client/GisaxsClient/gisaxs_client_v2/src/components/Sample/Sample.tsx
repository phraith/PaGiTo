import { Add } from "@mui/icons-material";
import {
  Button,
  Card,
  CardContent,
  Grid,
  List,
  ListItem,
  Typography,
} from "@mui/material";
import React, { useEffect } from "react";
import Layer from "./Layer";
import { v4 as uuidv4 } from "uuid";
import Substrate from "./Substrate";

interface SampleProps {
  jsonCallback: any;
}

const Sample = (props: SampleProps) => {
  const [layers, setLayers] = React.useState<any>([]);
  const [jsonData, setJsonData] = React.useState({});

  const removeLayer = (id: string) => {
    setLayers((layers) =>
      layers
        .filter((layer) => layer.props.id !== id)
        .map((value, index) => (
          <Layer
            key={value.props.id}
            id={value.props.id}
            order={index}
            
            removeCallback={() => removeLayer(value.props.id)}
          />
        ))
    );
  };

  useEffect(() => {
    props.jsonCallback(jsonData, "sample")
  }, [jsonData]);

  const addLayer = () => {
    const myid = uuidv4();
    setLayers([
      ...layers,
      <Layer
        key={myid}
        id={myid}
        order={layers.length}
        removeCallback={() => removeLayer(myid)}
      />,
    ]);
  };

  const jsonCallback = (value, key) => {
    jsonData[key] = value
    setJsonData({...jsonData})
  }

  return (
    <Card style={{ maxHeight: 700, overflow: "auto" }}>
      <CardContent>
        <Grid container rowSpacing={layers.length === 0 ? 0 : 2}>
          <Grid item xs={8}>
            <Typography>Sample</Typography>
          </Grid>
          <Grid
            item
            xs={4}
            sx={{
              paddingBottom: layers.length === 0 ? 2 : 0,
            }}
          >
            <Button size="small" onClick={addLayer}>
              <Add />
            </Button>
          </Grid>
          <Grid item xs={12}>
            <Substrate jsonCallback={jsonCallback}/>
          </Grid>
          <Grid item xs={12}>
            <List>
              {layers.map((value) => {
                return <ListItem>{value}</ListItem>;
              })}
            </List>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );
};

export default Sample;
