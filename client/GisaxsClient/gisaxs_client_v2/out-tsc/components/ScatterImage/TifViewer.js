"use strict";
// import { decode } from 'decode-tiff'
// import { useState } from 'react';
// import { encodeBase64 } from 'tweetnacl-util/nacl-util'
// const TifViewer = () => {
//     const [encoded, setEncoded] = useState<string>()
//     const handleFile = (e) => {
//         const content : string = e.target.result;
//         console.log('file content', content)
//         var data = [];
//         for (let i = 0; i < content.length; i++){  
//             data.push(content.charCodeAt(i));
//         }
//         console.log(data)
//         const { width, height, dataDecoded } = decode(data, );
//         let base64 = encodeBase64(new Uint8Array(dataDecoded.Buffer)) // Encode
//         setEncoded(base64)
//     }
//     const handleChangeFile = (file) => {
//         let fileData = new FileReader();
//         fileData.onloadend = handleFile;
//         fileData.readAsText(file);
//     }
//     // const { width, height, data } = decode();
//     // var encoded = encodeBase64(new Uint8Array(data.Buffer)) // Encode
//     return (<div>
//         <input type="file" onChange={e =>
//             handleChangeFile(e.target.files[0])} />
//         {/* <img
//             alt=""
//             style={{ height: "100%", width: "100%", display: 'block' }}
//             src={`data:image/jpeg;base64,${"encoded"}`}
//         /> */}
//     </div >
//     )
// }
// export default TifViewer
