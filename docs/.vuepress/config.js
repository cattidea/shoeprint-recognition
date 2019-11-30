module.exports = {
  title: "Shoeprint Recognition",
  description: "“恒锐杯”鞋印花纹图像类别判定挑战赛",
  base: "/shoeprint-recognition/",
  head: [["link", { rel: "icon", href: "/logo.svg" }]],

  markdown: {
    // markdown-it-anchor 的选项
    anchor: { permalink: false },
    // markdown-it-toc 的选项
    toc: { includeLevel: [1, 2, 3] },
    // markdown 插件
    extendMarkdown: md => {
      md.set({ breaks: true });
      md.use(require("@iktakahiro/markdown-it-katex"), {
        throwOnError: false,
        errorColor: " #cc0000"
      });
      md.use(require("markdown-it-mark"));
    }
  },

  // 插件
  plugins: [
    // 自动返回顶部
    "@vuepress/back-to-top",
    // 页面滚动时自动激活侧边栏链接
    "@vuepress/active-header-links",
    // 流程图插件
    "flowchart"
  ],

  // 主题配置
  themeConfig: {
    nav: [{ text: "主页", link: "/" }, { text: "指南", link: "/guide/" }],
    sidebarDepth: 1,
    sidebar: {
      "/guide/": ["", "usage", "io", "model", "faq", "notice"],
      "/": [""]
    },

    algolia: {
      apiKey: "4447290f7c449f572a53081c232ed21e",
      indexName: "siguremo_shoeprint-recognition"
    },

    serviceWorker: {
      updatePopup: true // Boolean | Object, 默认值是 undefined.
    },

    lastUpdated: "Last Updated", // string | boolean

    // 假定是 GitHub. 同时也可以是一个完整的 GitLab URL
    repo: "zsync/shoeprint-recognition",
    // 自定义仓库链接文字。默认从 `themeConfig.repo` 中自动推断为
    // "GitHub"/"GitLab"/"Bitbucket" 其中之一，或是 "Source"。
    repoLabel: "Github",

    // 以下为可选的编辑链接选项

    // 假如你的文档仓库和项目本身不在一个仓库：
    docsRepo: "zsync/shoeprint-recognition",
    // 假如文档不是放在仓库的根目录下：
    docsDir: "docs",
    // 假如文档放在一个特定的分支下：
    docsBranch: "master",
    // 默认是 false, 设置为 true 来启用
    editLinks: true,
    // 默认为 "Edit this page"
    editLinkText: "在GitHub上编辑此页！"
  }
};
