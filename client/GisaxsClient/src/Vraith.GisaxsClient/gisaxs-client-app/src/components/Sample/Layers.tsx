// import React, { useEffect } from "react";
// import { LayerConfig, SetLocalStorageEntity } from "../Utility/DefaultConfigs";
// import { v4 as uuidv4 } from "uuid";
// import Layer from "./Layer";
// import Button from "@mui/material/Button"
// import Card from "@mui/material/Card"
// import CardActions from "@mui/material/CardActions"
// import CardContent from "@mui/material/CardContent"
// import Collapse from "@mui/material/Collapse"
// import FormControl from "@mui/material/FormControl"
// import Grid from "@mui/material/Grid"
// import Typography from "@mui/material/Typography"
// import TextField from "@mui/material/TextField"
// import Stack from "@mui/material/Stack"
// import  Add from "@mui/icons-material/Add";
// import List from "@mui/material/List"
// import ListItem from "@mui/material/ListItem"
// interface LayersProps {
//   jsonCallback: any;
// }

// const Layers = (props: LayersProps) => {
//   const [layers, setLayers] = React.useState<any>([]);
//   const [jsonData, setJsonData] = React.useState({});

//   const localStorageEntityName: string = "layersConfig";
//   const configFieldName: string = "layers";

//   useEffect(() => {
//     let formattedLayers = Object.keys(jsonData).map((key) => jsonData[key]);

//     props.jsonCallback(formattedLayers, configFieldName);
//     console.log("Set layersConfig");
//     console.log(formattedLayers);
//     SetLocalStorageEntity(formattedLayers, [], localStorageEntityName);
//   }, [jsonData]);

//   useEffect(() => {
//     let data = localStorage.getItem(localStorageEntityName);
//     console.log(data);
//     if (data !== null) {
//       let layersConfig = JSON.parse(data);
//       let cachedLayers: any = [];
//       for (var layer of layersConfig) {
//         console.log(layer);
//         cachedLayers = [...cachedLayers, createLayer(layer)];
//       }
//       setLayers(cachedLayers);
//     }
//   }, []);

//   const removeLayer = (id: string) => {
//     setLayers((layers) => layers.filter((layer) => layer.props.id !== id));
//     setJsonData((jsonData) => {
//       delete jsonData[id];
//       return { ...jsonData };
//     });
//   };

//   const createJsonForLayer = (layerJson, layerId) => {
//     setJsonData((jsonData) => {
//       jsonData[layerId] = layerJson;
//       return { ...jsonData };
//     });
//   };

//   const createLayer = (layerConfig) => {
//     const myid = uuidv4();
//     return (
//       <Layer
//         key={myid}
//         id={myid}
//         order={layerConfig.order == -1 ? layers.length : layerConfig.order}
//         removeCallback={() => removeLayer(myid)}
//         initialConfig={layerConfig}
//         jsonCallback={createJsonForLayer}
//       />
//     );
//   };

//   const addLayer = () => {
//     setLayers([...layers, createLayer(LayerConfig)]);
//   };

//   return (
//     <Grid container>
//       <Grid item xs={12}>
//         <List>
//           {layers.map((value) => {
//             return <ListItem key={value.props.id}>{value}</ListItem>;
//           })}
//         </List>
//       </Grid>
//       <Grid
//         item
//         xs={4}
//         sx={{
//           paddingBottom: layers.length === 0 ? 2 : 0,
//         }}
//       >
//         <Button size="small" onClick={addLayer}>
//           <Add />
//         </Button>
//       </Grid>
//     </Grid>
//   );
// };

// export default Layers;
