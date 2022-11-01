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
// import React, { useEffect } from "react";
// import Substrate from "./Substrate";
// import Layers from "./Layers";

// interface SampleProps {
//   jsonCallback: any;
// }

// const SampleTemp = (props: SampleProps) => {
//   const [jsonData, setJsonData] = React.useState({
//     layers: {}
//   });

//   const configFieldName: string = "sample";

//   useEffect(() => {
//     props.jsonCallback(jsonData, configFieldName);
//   }, [jsonData]);

//   const jsonCallback = (value, key) => {
//     jsonData[key] = value;
//     setJsonData({ ...jsonData });
//   };

//   return (
//     <Card style={{ maxHeight: 700, overflow: "auto" }}>
//       <CardContent>
//         <Grid container rowSpacing={2}>
//           <Grid item xs={8}>
//             <Typography>Sample</Typography>
//           </Grid>
//           <Grid item xs={12}>
//             <Substrate jsonCallback={jsonCallback} />
//           </Grid>
//           <Grid item xs={12}>
//             <Layers jsonCallback={jsonCallback}/>
//           </Grid>
//         </Grid>
//       </CardContent>
//     </Card>
//   );
// };

// export default SampleTemp;
