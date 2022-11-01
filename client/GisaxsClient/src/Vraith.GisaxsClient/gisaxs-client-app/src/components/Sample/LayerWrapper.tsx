// import { DeleteForever, ExpandLess, ExpandMore } from "@mui/icons-material";
// import { Box, Button, Card, CardContent, Collapse, Typography } from "@mui/material";
// import { useState } from "react";

// interface LayerWrapperProps {
//     removeCallback: any;
//     children: React.ReactNode;
// }

// const LayerWrapper: React.FC<LayerWrapperProps> = (props: LayerWrapperProps) => {

//     return (
//         <Card sx={{}}>
//             <CardContent>
//                 <Box display="flex" sx={{ justifyContent: "space-between" }}>
//                     <Button size="small" onClick={handleButtonClick}>
//                         {collapsed ? <ExpandMore /> : <ExpandLess />}
//                     </Button>
//                     <Typography sx={{ fontSize: 14 }} color="text.secondary" gutterBottom>
//                         Layer
//                     </Typography>

//                     <Button size="small" onClick={handleRemove}>
//                         <DeleteForever />
//                     </Button>
//                 </Box>
//                 <Collapse in={!collapsed}>
//                     {props.children}
//                 </Collapse>
//             </CardContent>
//         </Card>
//     );
// }

// export default LayerWrapper