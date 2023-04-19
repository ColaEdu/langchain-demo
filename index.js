// use common js to import the module
const { OpenAI } = require("langchain/llms/openai");
const dotenv = require("dotenv");
const { HttpsProxyAgent } = require("https-proxy-agent");

dotenv.config();

const express = require("express");
const { createProxyMiddleware } = require("http-proxy-middleware");
async function main() {
  const openAIApiKey = process.env.OPENAI_API_KEY;
  const model = new OpenAI({ openAIApiKey: openAIApiKey, temperature: 0.9 });
  const res = await model.call(
    "What would be a good company name a company that makes colorful socks?"
  );
  console.log(res);
}

main()